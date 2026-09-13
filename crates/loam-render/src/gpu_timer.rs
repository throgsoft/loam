//! wgpu rejects `MAP_READ | QUERY_RESOLVE` on one buffer and locks a whole
//! buffer while any slice is mapped, so one resolve buffer feeds a map buffer
//! per slot.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, Features, MapMode, QuerySet,
    QuerySetDescriptor, QueryType, Queue,
};

const MAX_SECTIONS: usize = 16;

const SECTION_BYTES: u64 = 16;

/// Sixteen sections per frame; a result arrives a frame late, and a frame whose map is still in flight measures nothing.
pub struct SectionTimer {
    query_set: QuerySet,
    resolve_buffer: Buffer,
    map_buffer: Buffer,
    timestamp_period_ns: f32,
    open: usize,
    resolved: Option<usize>,
    names: [&'static str; MAX_SECTIONS],
    in_flight: Arc<AtomicBool>,
    last: Arc<Mutex<SectionResults>>,
}

struct SectionResults {
    names: [&'static str; MAX_SECTIONS],
    elapsed: [Duration; MAX_SECTIONS],
}

impl SectionTimer {
    pub fn new(device: &Device, queue: &Queue) -> Option<Self> {
        let needed = Features::TIMESTAMP_QUERY | Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        if !device.features().contains(needed) {
            return None;
        }
        let query_set = device.create_query_set(&QuerySetDescriptor {
            label: Some("loam-render::SectionTimer::query_set"),
            ty: QueryType::Timestamp,
            count: (MAX_SECTIONS * 2) as u32,
        });
        let resolve_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("loam-render::SectionTimer::resolve_buffer"),
            size: SECTION_BYTES * MAX_SECTIONS as u64,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let map_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("loam-render::SectionTimer::map_buffer"),
            size: SECTION_BYTES * MAX_SECTIONS as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Some(Self {
            query_set,
            resolve_buffer,
            map_buffer,
            timestamp_period_ns: queue.get_timestamp_period(),
            open: 0,
            resolved: None,
            names: [""; MAX_SECTIONS],
            in_flight: Arc::new(AtomicBool::new(false)),
            last: Arc::new(Mutex::new(SectionResults {
                names: [""; MAX_SECTIONS],
                elapsed: [Duration::ZERO; MAX_SECTIONS],
            })),
        })
    }

    pub fn begin_frame(&mut self) {
        self.open = 0;
    }

    pub fn open(&mut self, encoder: &mut CommandEncoder, name: &'static str) -> Option<usize> {
        if self.open >= MAX_SECTIONS {
            return None;
        }
        let slot = self.open;
        encoder.write_timestamp(&self.query_set, (slot * 2) as u32);
        self.names[slot] = name;
        self.open += 1;
        Some(slot)
    }

    pub fn close(&mut self, encoder: &mut CommandEncoder, slot: usize) {
        encoder.write_timestamp(&self.query_set, (slot * 2 + 1) as u32);
    }

    /// The last mapped result for `slot` while its section keeps the same name.
    pub fn elapsed(&self, slot: usize) -> Option<Duration> {
        let last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        (last.names[slot] == self.names[slot]).then(|| last.elapsed[slot])
    }

    pub fn resolve(&mut self, encoder: &mut CommandEncoder) {
        if self.open == 0 || self.in_flight.load(Ordering::Acquire) {
            return;
        }
        let bytes = SECTION_BYTES * self.open as u64;
        encoder.resolve_query_set(
            &self.query_set,
            0..(self.open * 2) as u32,
            &self.resolve_buffer,
            0,
        );
        encoder.copy_buffer_to_buffer(&self.resolve_buffer, 0, &self.map_buffer, 0, bytes);
        self.in_flight.store(true, Ordering::Release);
        self.resolved = Some(self.open);
    }

    pub fn after_submit(&mut self) -> bool {
        let Some(open) = self.resolved.take() else {
            return false;
        };
        let bytes = SECTION_BYTES * open as u64;
        let names = self.names;
        let buffer = self.map_buffer.clone();
        let reader = buffer.clone();
        let period_ns = self.timestamp_period_ns;
        let last = self.last.clone();
        let flag = self.in_flight.clone();
        buffer
            .slice(0..bytes)
            .map_async(MapMode::Read, move |result| {
                if result.is_ok() {
                    let view = reader.slice(0..bytes).get_mapped_range();
                    let mut done = last.lock().unwrap_or_else(|e| e.into_inner());
                    done.names = [""; MAX_SECTIONS];
                    for (slot, name) in names.iter().enumerate().take(open) {
                        let at = slot * SECTION_BYTES as usize;
                        let (Ok(start), Ok(end)) = (
                            <[u8; 8]>::try_from(&view[at..at + 8]),
                            <[u8; 8]>::try_from(&view[at + 8..at + 16]),
                        ) else {
                            break;
                        };
                        let ticks =
                            u64::from_le_bytes(end).saturating_sub(u64::from_le_bytes(start));
                        let nanos = (ticks as f64 * period_ns as f64) as u64;
                        done.names[slot] = name;
                        done.elapsed[slot] = Duration::from_nanos(nanos);
                    }
                    drop(done);
                    drop(view);
                    reader.unmap();
                }
                flag.store(false, Ordering::Release);
            });
        true
    }
}
