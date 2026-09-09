use crate::command::RequestId;
use crate::phase::{EntryId, Readback, Schedule, Tick};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BulkId(u32);

impl BulkId {
    pub(crate) fn new(index: usize) -> Self {
        Self(index as u32)
    }

    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapshotPolicy {
    /// Restored from its checkpoint; a snapshot needs the checkpoint at the current tick and a restore refuses without one.
    Authoritative,
    /// Zeroed on restore and after a device loss; the next work item fills it.
    Reinitializable,
    /// Left alone on restore; its items recompute it from what they read.
    Derived,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BulkSpec {
    pub name: &'static str,
    pub element_size: u32,
    pub count: u32,
    pub readback: Readback,
    pub snapshot: SnapshotPolicy,
    pub schedule: Schedule,
}

impl BulkSpec {
    pub fn bytes(&self) -> u64 {
        self.element_size as u64 * self.count as u64
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BulkError {
    Unknown(BulkId),
    Removed(BulkId),
}

#[derive(Default)]
pub struct Bulk {
    specs: Vec<BulkSpec>,
    live: Vec<bool>,
    grown: usize,
}

impl Bulk {
    pub(crate) fn register(&mut self, spec: BulkSpec) -> BulkId {
        self.grown += spec.count as usize;
        self.specs.push(spec);
        self.live.push(true);
        BulkId((self.specs.len() - 1) as u32)
    }

    pub(crate) fn remove(&mut self, id: BulkId) -> bool {
        match self.live.get_mut(id.index()) {
            Some(live) if *live => {
                *live = false;
                true
            }
            _ => false,
        }
    }

    pub(crate) fn take_growth(&mut self) -> usize {
        std::mem::take(&mut self.grown)
    }

    pub(crate) fn len(&self) -> usize {
        self.specs.len()
    }

    pub(crate) fn is_live(&self, id: BulkId) -> bool {
        self.live.get(id.index()).copied().unwrap_or(false)
    }

    /// Live stores only; a removed id keeps its index.
    pub fn iter(&self) -> impl Iterator<Item = (BulkId, &BulkSpec)> {
        self.specs
            .iter()
            .enumerate()
            .filter(|(index, _)| self.live[*index])
            .map(|(index, spec)| (BulkId(index as u32), spec))
    }

    pub(crate) fn snapshot(&self, checkpoints: &[Option<BulkCheckpoint>]) -> BulkSnapshot {
        BulkSnapshot {
            specs: self.specs.clone(),
            live: self.live.clone(),
            checkpoints: checkpoints.to_vec(),
        }
    }

    pub(crate) fn restore(&mut self, from: &BulkSnapshot) {
        self.specs.clear();
        self.specs.extend_from_slice(&from.specs);
        self.live.clear();
        self.live.extend_from_slice(&from.live);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BulkCheckpoint {
    pub tick: Tick,
    pub rows: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BulkAction {
    Replace,
    Reinitialize,
}

pub struct BulkSnapshot {
    pub(crate) specs: Vec<BulkSpec>,
    pub(crate) live: Vec<bool>,
    pub(crate) checkpoints: Vec<Option<BulkCheckpoint>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkOrder {
    pub entry: EntryId,
    pub name: &'static str,
    pub schedule: Schedule,
    pub readback: Readback,
    pub tick: Tick,
    pub request: RequestId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Landing {
    Applied,
    /// Landed after a reset, a restore, a cancellation, or the store's removal; nothing was applied.
    Discarded,
}

pub struct Landed<'a> {
    pub work: &'static str,
    pub request: RequestId,
    pub tick: Tick,
    pub rows: &'a [u8],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkStats {
    pub issued: u64,
    pub fallbacks: u64,
    pub delayed: u64,
    pub discarded: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wait {
    pub entry: EntryId,
    pub work: &'static str,
    pub request: RequestId,
    pub tick: Tick,
}

#[derive(Default)]
struct Slot {
    order: Option<WorkOrder>,
    landed: bool,
    writes: Vec<BulkId>,
    rows: Vec<u8>,
}

impl Slot {
    fn clear(&mut self) {
        self.order = None;
        self.landed = false;
        self.writes.clear();
        self.rows.clear();
    }
}

pub(crate) struct InFlight {
    slots: Vec<Slot>,
    discarded: u64,
}

impl InFlight {
    pub(crate) fn new(capacity: u32) -> Self {
        Self {
            slots: (0..capacity.max(1)).map(|_| Slot::default()).collect(),
            discarded: 0,
        }
    }

    pub(crate) fn discarded(&self) -> u64 {
        self.discarded
    }

    pub(crate) fn submit(&mut self, order: WorkOrder, writes: &[BulkId]) -> bool {
        let Some(slot) = self.slots.iter_mut().find(|slot| slot.order.is_none()) else {
            return false;
        };
        slot.order = Some(order);
        slot.landed = false;
        slot.writes.clear();
        slot.writes.extend_from_slice(writes);
        slot.rows.clear();
        true
    }

    pub(crate) fn land(&mut self, request: RequestId, rows: &[u8]) -> Landing {
        let Some(slot) = self
            .slots
            .iter_mut()
            .find(|slot| slot.order.is_some_and(|order| order.request == request))
        else {
            self.discarded += 1;
            return Landing::Discarded;
        };
        if slot
            .order
            .is_some_and(|order| order.readback == Readback::None)
        {
            slot.clear();
            return Landing::Applied;
        }
        slot.landed = true;
        slot.rows.clear();
        slot.rows.extend_from_slice(rows);
        Landing::Applied
    }

    pub(crate) fn landed(&self) -> impl Iterator<Item = Landed<'_>> {
        self.slots
            .iter()
            .filter(|slot| slot.landed)
            .filter_map(|slot| {
                let order = slot.order?;
                Some(Landed {
                    work: order.name,
                    request: order.request,
                    tick: order.tick,
                    rows: &slot.rows,
                })
            })
    }

    pub(crate) fn release(&mut self, request: RequestId) -> bool {
        match self
            .slots
            .iter_mut()
            .find(|slot| slot.order.is_some_and(|order| order.request == request))
        {
            Some(slot) => {
                slot.clear();
                true
            }
            None => false,
        }
    }

    pub(crate) fn outstanding(&self, work: &str) -> Option<WorkOrder> {
        self.slots
            .iter()
            .filter(|slot| !slot.landed)
            .filter_map(|slot| slot.order)
            .find(|order| order.name == work && order.readback == Readback::Required)
    }

    pub(crate) fn any_required(&self) -> Option<WorkOrder> {
        self.slots
            .iter()
            .filter(|slot| !slot.landed)
            .filter_map(|slot| slot.order)
            .find(|order| order.readback == Readback::Required)
    }

    pub(crate) fn writes_pending(&self, id: BulkId) -> bool {
        self.slots
            .iter()
            .any(|slot| !slot.landed && slot.order.is_some() && slot.writes.contains(&id))
    }

    pub(crate) fn cancel_all(&mut self) {
        for slot in &mut self.slots {
            slot.clear();
        }
    }

    pub(crate) fn cancel_bulk(&mut self, id: BulkId) {
        for slot in &mut self.slots {
            if slot.writes.contains(&id) {
                slot.clear();
            }
        }
    }
}
