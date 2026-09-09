use std::sync::{Arc, Mutex};

use loam_runtime::{BulkId, BulkSpec, RequestId};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, Buffer, BufferDescriptor, BufferUsages, CommandEncoder,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device, MapMode, PollType,
    Queue, ShaderModuleDescriptor, ShaderSource,
};

const BULK_USAGES: BufferUsages = BufferUsages::STORAGE
    .union(BufferUsages::VERTEX)
    .union(BufferUsages::COPY_SRC)
    .union(BufferUsages::COPY_DST);

#[derive(Default)]
pub struct BulkBuffers {
    buffers: Vec<Option<Buffer>>,
    zeros: Vec<u8>,
}

impl BulkBuffers {
    /// Creates or regrows the buffer; one already large enough keeps its contents.
    pub fn ensure(&mut self, device: &Device, id: BulkId, spec: &BulkSpec) {
        if self.buffers.len() <= id.index() {
            self.buffers.resize_with(id.index() + 1, || None);
        }
        let size = spec.bytes().max(spec.element_size as u64);
        let stale = self.buffers[id.index()]
            .as_ref()
            .is_none_or(|buffer| buffer.size() < size);
        if stale {
            self.buffers[id.index()] = Some(device.create_buffer(&BufferDescriptor {
                label: Some(spec.name),
                size,
                usage: BULK_USAGES,
                mapped_at_creation: false,
            }));
        }
    }

    pub fn get(&self, id: BulkId) -> Option<&Buffer> {
        self.buffers.get(id.index())?.as_ref()
    }

    pub fn remove(&mut self, id: BulkId) {
        if let Some(slot) = self.buffers.get_mut(id.index()) {
            *slot = None;
        }
    }

    pub fn write(&self, queue: &Queue, id: BulkId, rows: &[u8]) {
        if let Some(buffer) = self.get(id) {
            queue.write_buffer(buffer, 0, &rows[..rows.len().min(buffer.size() as usize)]);
        }
    }

    pub fn reinitialize(&mut self, queue: &Queue, id: BulkId) {
        let Some(size) = self.get(id).map(|buffer| buffer.size() as usize) else {
            return;
        };
        if self.zeros.len() < size {
            self.zeros.resize(size, 0);
        }
        self.write(queue, id, &self.zeros[..size]);
    }
}

pub struct ComputeWork {
    pipeline: ComputePipeline,
    bind: wgpu::BindGroup,
}

impl ComputeWork {
    /// `buffers` bind to group 0 in order under the layout the shader declares.
    pub fn new(
        device: &Device,
        label: &'static str,
        wgsl: &str,
        entry: &str,
        buffers: &[&Buffer],
    ) -> Self {
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some(label),
            source: ShaderSource::Wgsl(wgsl.into()),
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        let entries: Vec<BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some(label),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        Self { pipeline, bind }
    }

    pub fn record(&self, encoder: &mut CommandEncoder, label: &'static str, workgroups: u32) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind, &[]);
        pass.dispatch_workgroups(workgroups.max(1), 1, 1);
    }
}

type Completion = Arc<Mutex<Option<bool>>>;

struct Pending {
    request: RequestId,
    staging: Buffer,
    size: u64,
    mapped: bool,
    done: Completion,
}

#[derive(Default)]
pub struct Readbacks {
    pending: Vec<Pending>,
    idle: Vec<Buffer>,
    rows: Vec<u8>,
}

impl Readbacks {
    pub fn request(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        request: RequestId,
        source: &Buffer,
        size: u64,
    ) {
        let size = size.min(source.size());
        if size == 0 {
            return;
        }
        let staging = match self.idle.iter().position(|buffer| buffer.size() >= size) {
            Some(index) => self.idle.swap_remove(index),
            None => device.create_buffer(&BufferDescriptor {
                label: Some("bulk readback"),
                size,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        };
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size);
        self.pending.push(Pending {
            request,
            staging,
            size,
            mapped: false,
            done: Arc::new(Mutex::new(None)),
        });
    }

    pub fn after_submit(&mut self) {
        for pending in self.pending.iter_mut().filter(|pending| !pending.mapped) {
            pending.mapped = true;
            let signal = pending.done.clone();
            pending
                .staging
                .slice(..pending.size)
                .map_async(MapMode::Read, move |result| {
                    *signal.lock().unwrap_or_else(|error| error.into_inner()) =
                        Some(result.is_ok());
                });
        }
    }

    /// Never blocks; delivers the copies whose maps have completed by this poll.
    pub fn poll(
        &mut self,
        device: &Device,
        mut deliver: impl FnMut(RequestId, Option<&[u8]>),
    ) -> usize {
        let _ = device.poll(PollType::Poll);
        let mut delivered = 0;
        let mut index = 0;
        while index < self.pending.len() {
            let done = match self.pending[index].mapped {
                true => *self.pending[index]
                    .done
                    .lock()
                    .unwrap_or_else(|error| error.into_inner()),
                false => None,
            };
            let Some(mapped) = done else {
                index += 1;
                continue;
            };
            let pending = self.pending.swap_remove(index);
            delivered += 1;
            if !mapped {
                deliver(pending.request, None);
                continue;
            }
            self.rows.clear();
            self.rows
                .extend_from_slice(&pending.staging.slice(..pending.size).get_mapped_range());
            pending.staging.unmap();
            deliver(pending.request, Some(&self.rows));
            self.idle.push(pending.staging);
        }
        delivered
    }

    pub fn cancel(&mut self) {
        self.pending.clear();
        self.idle.clear();
    }
}

#[cfg(test)]
mod tests {
    use loam_runtime::{
        BulkSpec, Input, Landing, Phase, Readback, Schedule, Session, SimConfig, SnapshotPolicy,
        WorkItem,
    };

    use super::*;

    loam_runtime::stores! {
        #[derive(Default)]
        pub struct Bare {}
    }

    #[test]
    fn a_failed_map_stays_pending_and_never_retires_its_readback() {
        let gpu = crate::device::noop_context();
        let mut session = Session::new(Bare::default(), SimConfig::default());
        let grid = session.register_bulk(BulkSpec {
            name: "grid",
            element_size: 4,
            count: 4,
            readback: Readback::Required,
            snapshot: SnapshotPolicy::Derived,
            schedule: Schedule::InStep,
        });
        session.work(
            Phase::Simulation,
            WorkItem::new("reduce", Schedule::InStep, Readback::Required).writes(grid),
        );
        session.boundary(Input::default()).expect("boundary");
        session.tick().expect("tick");
        let mut request = None;
        session.issue_work(|order| request = Some(order.request));
        let request = request.expect("the tick ordered its work item");

        let mut readbacks = Readbacks::default();
        readbacks.pending.push(Pending {
            request,
            staging: gpu.device.create_buffer(&BufferDescriptor {
                label: None,
                size: 16,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            size: 16,
            mapped: true,
            done: Arc::new(Mutex::new(Some(false))),
        });

        let mut landings = Vec::new();
        let retired = readbacks.poll(&gpu.device, |request, rows| {
            landings.push(session.land_readback(request, rows));
        });

        assert_eq!(retired, 1);
        assert!(readbacks.pending.is_empty());
        assert!(readbacks.idle.is_empty());
        assert_eq!(landings, [Landing::Failed]);
        assert_eq!(session.readbacks().count(), 0);
        assert_eq!(session.waiting(), None);
    }
}
