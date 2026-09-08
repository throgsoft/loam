//! wgpu rejects `MAP_READ | QUERY_RESOLVE` on one buffer and locks a whole
//! buffer while any slice is mapped, so one resolve buffer feeds a map buffer
//! per slot.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Arc;
use std::time::Duration;
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, Features, MapMode, QuerySet,
    QuerySetDescriptor, QueryType, Queue, QUERY_RESOLVE_BUFFER_ALIGNMENT,
};

const FRAMES_IN_FLIGHT: usize = 3;

const BYTES_PER_SLOT: u64 = 16;

const SLOT_STRIDE_BYTES: u64 = QUERY_RESOLVE_BUFFER_ALIGNMENT;

struct SlotState {
    in_flight: Arc<AtomicBool>,
    map_buffer: Buffer,
}

pub struct GpuTimer {
    query_set: QuerySet,
    resolve_buffer: Buffer,
    slots: [SlotState; FRAMES_IN_FLIGHT],
    frame_index: u64,
    timestamp_period_ns: f32,
    rx: Receiver<Duration>,
    tx: SyncSender<Duration>,
    started_slot: Cell<Option<usize>>,
    resolved_slot: Cell<Option<usize>>,
}

impl GpuTimer {
    pub fn new(device: &Device, queue: &Queue) -> Option<Self> {
        let needed = Features::TIMESTAMP_QUERY | Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        if !device.features().contains(needed) {
            return None;
        }
        let query_set = device.create_query_set(&QuerySetDescriptor {
            label: Some("loam-render::GpuTimer::query_set"),
            ty: QueryType::Timestamp,
            count: (FRAMES_IN_FLIGHT * 2) as u32,
        });
        let resolve_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("loam-render::GpuTimer::resolve_buffer"),
            size: SLOT_STRIDE_BYTES * FRAMES_IN_FLIGHT as u64,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let slots = std::array::from_fn(|_| SlotState {
            in_flight: Arc::new(AtomicBool::new(false)),
            map_buffer: device.create_buffer(&BufferDescriptor {
                label: Some("loam-render::GpuTimer::map_buffer"),
                size: BYTES_PER_SLOT,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        });
        let (tx, rx) = sync_channel(FRAMES_IN_FLIGHT);
        Some(Self {
            query_set,
            resolve_buffer,
            slots,
            frame_index: 0,
            timestamp_period_ns: queue.get_timestamp_period(),
            rx,
            tx,
            started_slot: Cell::new(None),
            resolved_slot: Cell::new(None),
        })
    }

    fn current_slot(&self) -> usize {
        (self.frame_index as usize) % FRAMES_IN_FLIGHT
    }

    fn slot_query_range(slot: usize) -> std::ops::Range<u32> {
        let base = (slot * 2) as u32;
        base..(base + 2)
    }

    fn slot_byte_range(slot: usize) -> std::ops::Range<u64> {
        let base = slot as u64 * SLOT_STRIDE_BYTES;
        base..(base + BYTES_PER_SLOT)
    }

    pub fn write_start(&self, encoder: &mut CommandEncoder) {
        self.started_slot.set(None);
        let slot = self.current_slot();
        if self.slots[slot].in_flight.load(Ordering::Acquire) {
            return;
        }
        let range = Self::slot_query_range(slot);
        encoder.write_timestamp(&self.query_set, range.start);
        self.started_slot.set(Some(slot));
    }

    pub fn write_end_and_resolve(&self, encoder: &mut CommandEncoder) {
        let Some(slot) = self.started_slot.take() else {
            return;
        };
        let query_range = Self::slot_query_range(slot);
        let byte_range = Self::slot_byte_range(slot);
        encoder.write_timestamp(&self.query_set, query_range.end - 1);
        encoder.resolve_query_set(
            &self.query_set,
            query_range,
            &self.resolve_buffer,
            byte_range.start,
        );
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            byte_range.start,
            &self.slots[slot].map_buffer,
            0,
            BYTES_PER_SLOT,
        );
        self.slots[slot].in_flight.store(true, Ordering::Release);
        self.resolved_slot.set(Some(slot));
    }

    /// Call once per redraw, after the end-of-frame queue submit.
    pub fn tick(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);

        while let Ok(duration) = self.rx.try_recv() {
            loam_time::frame_trace::record_external("gpu-total", duration);
        }

        let Some(just_resolved_slot) = self.resolved_slot.take() else {
            return;
        };
        let buffer = self.slots[just_resolved_slot].map_buffer.clone();
        let buffer_for_callback = buffer.clone();
        let period_ns = self.timestamp_period_ns;
        let tx = self.tx.clone();
        let flag = self.slots[just_resolved_slot].in_flight.clone();
        buffer.slice(..).map_async(MapMode::Read, move |result| {
            if result.is_ok() {
                let view = buffer_for_callback.slice(..).get_mapped_range();
                if let (Ok(start_bytes), Ok(end_bytes)) = (
                    <[u8; 8]>::try_from(&view[0..8]),
                    <[u8; 8]>::try_from(&view[8..16]),
                ) {
                    let start_ticks = u64::from_le_bytes(start_bytes);
                    let end_ticks = u64::from_le_bytes(end_bytes);
                    if let Some(delta_ticks) = end_ticks.checked_sub(start_ticks) {
                        let delta_ns = (delta_ticks as f64 * period_ns as f64) as u64;
                        let _ = tx.try_send(Duration::from_nanos(delta_ns));
                    }
                }
                drop(view);
                buffer_for_callback.unmap();
            }
            flag.store(false, Ordering::Release);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const _: () = assert!(SLOT_STRIDE_BYTES >= BYTES_PER_SLOT);

    #[test]
    fn slot_byte_range_is_aligned_and_disjoint() {
        for slot in 0..FRAMES_IN_FLIGHT {
            let range = GpuTimer::slot_byte_range(slot);
            assert_eq!(
                range.start % QUERY_RESOLVE_BUFFER_ALIGNMENT,
                0,
                "slot {slot} start not aligned"
            );
            assert_eq!(range.end - range.start, BYTES_PER_SLOT);
        }
        for slot in 0..FRAMES_IN_FLIGHT.saturating_sub(1) {
            let a = GpuTimer::slot_byte_range(slot);
            let b = GpuTimer::slot_byte_range(slot + 1);
            assert!(a.end <= b.start);
        }
    }
}
