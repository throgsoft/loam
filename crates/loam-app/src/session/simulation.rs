#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::{self, Receiver, Sender};
#[cfg(not(target_arch = "wasm32"))]
use std::thread::{self, JoinHandle};

use std::time::Duration;

use web_time::Instant;

use loam_runtime::host::HostError;
use loam_runtime::{ActionId, CursorPolicy, Eye, Input, PhaseError, Publication, Session, Stores};
use loam_time::par::Executor;
use loam_time::{frame_trace, FixedTimestep};

use super::app::SessionApp;
use super::commands::CommandInbox;
use super::frame::failed;
use super::input::InputMap;

struct Request<P> {
    now: Instant,
    aspect: f32,
    input: Input,
    returned: Option<Publication<P>>,
    restart: Option<Instant>,
}

pub(crate) struct Prepared<P> {
    pub(crate) publication: Publication<P>,
    pub(crate) eye: Eye,
    pub(crate) cursor: Option<CursorPolicy>,
}

pub(crate) enum Failure {
    Faulted { error: HostError, fresh: bool },
    Fatal(HostError),
}

pub(crate) struct Reply<P> {
    pub(crate) input: Input,
    pub(crate) recovered: Option<PhaseError>,
    pub(crate) outcome: Result<Prepared<P>, Failure>,
    spent: Spent,
}

#[derive(Default)]
struct Spent {
    dispatch: Option<Duration>,
    simulation: Option<Duration>,
    publication: Option<Duration>,
}

impl Spent {
    fn record(&self, [dispatch, simulation, publication]: [&'static str; 3]) {
        let sections = [
            (dispatch, self.dispatch),
            (simulation, self.simulation),
            (publication, self.publication),
        ];
        for (name, elapsed) in sections {
            if let Some(elapsed) = elapsed {
                frame_trace::record_external(name, elapsed);
            }
        }
    }
}

pub(crate) struct Simulation<A: Stores> {
    session: Session<A>,
    executor: Box<dyn Executor>,
    timestep: Option<FixedTimestep>,
    inbox: CommandInbox<A>,
    fault_recovery: Option<ActionId>,
    idle: Option<Publication<A::Published>>,
}

impl<A: Stores> Simulation<A> {
    pub(crate) fn new(
        session: Session<A>,
        app: &SessionApp<A>,
        executor: Box<dyn Executor>,
    ) -> Result<Self, HostError> {
        let sim = session.config();
        sim.check().map_err(failed)?;
        let timestep = sim
            .dt()
            .map(|_| FixedTimestep::new(sim.fixed_hz).with_max_catch_up(sim.max_ticks_per_frame));
        Ok(Self {
            session,
            executor,
            timestep,
            inbox: app.commands.clone(),
            fault_recovery: app.fault_recovery,
            idle: None,
        })
    }

    fn prepare(
        &mut self,
        request: Request<A::Published>,
        before: impl FnOnce(&Session<A>, &Input),
    ) -> Reply<A::Published> {
        let Request {
            now,
            aspect,
            input,
            returned,
            restart,
        } = request;
        if let (Some(at), Some(timestep)) = (restart, self.timestep.as_mut()) {
            timestep.reset_clock(at);
        }
        if returned.is_some() {
            self.idle = returned;
        }
        let faulted_before = self.session.faulted_phase();
        let scene_before = self.session.scene();
        let mut spent = Spent::default();
        let (recovered, advanced) = self.advance(now, aspect, input, before, &mut spent);
        let input = self.session.take_input();
        let outcome = match advanced.and_then(|()| self.publish(aspect, &mut spent)) {
            Ok(prepared) => Ok(prepared),
            Err(error) if self.session.faulted_phase().is_some() => {
                if let Some(timestep) = self.timestep.as_mut() {
                    timestep.reset_clock(now);
                }
                let fresh = faulted_before.is_none() || self.session.scene() != scene_before;
                Err(Failure::Faulted { error, fresh })
            }
            Err(error) => Err(Failure::Fatal(error)),
        };
        Reply {
            input,
            recovered,
            outcome,
            spent,
        }
    }

    fn advance(
        &mut self,
        now: Instant,
        aspect: f32,
        gathered: Input,
        before: impl FnOnce(&Session<A>, &Input),
        spent: &mut Spent,
    ) -> (Option<PhaseError>, Result<(), HostError>) {
        let Simulation {
            session,
            executor,
            timestep,
            inbox,
            fault_recovery,
            ..
        } = self;
        session.views_mut().root_mut().eye.aspect = aspect;
        let was_faulted = session.faulted_phase().is_some();
        if fault_recovery.is_some_and(|action| was_faulted && gathered.pressed(action)) {
            inbox.sender().reset();
        }
        before(session, &gathered);
        let started = Instant::now();
        let (recovered, dispatched) = inbox.boundary(session, gathered);
        spent.dispatch = Some(started.elapsed());
        let result = dispatched.and_then(|_| {
            session.views_mut().root_mut().eye.aspect = aspect;
            if let Some(timestep) = timestep.as_mut() {
                if was_faulted && session.faulted_phase().is_none() {
                    timestep.reset_clock(now);
                }
                let started = Instant::now();
                let ticked = timestep
                    .advance(now)
                    .try_for_each(|_| session.tick_with(executor.as_ref()));
                spent.simulation = Some(started.elapsed());
                ticked?;
            }
            session.views_mut().root_mut().eye.aspect = aspect;
            Ok(())
        });
        (recovered, result)
    }

    fn publish(
        &mut self,
        aspect: f32,
        spent: &mut Spent,
    ) -> Result<Prepared<A::Published>, HostError> {
        let mut publication = self.idle.take().unwrap_or_default();
        let alpha = self.timestep.as_ref().map_or(1.0, FixedTimestep::alpha);
        let started = Instant::now();
        let published = self
            .session
            .publish_with(&mut publication, alpha, self.executor.as_ref());
        spent.publication = Some(started.elapsed());
        if let Err(error) = published {
            self.idle = Some(publication);
            return Err(HostError::Phase(error));
        }
        self.session.views_mut().root_mut().eye.aspect = aspect;
        let views = self.session.views();
        let Some(root) = views.get(views.root()) else {
            self.idle = Some(publication);
            return Err(HostError::Host("the root view is missing".into()));
        };
        Ok(Prepared {
            eye: root.eye,
            cursor: root.rig.cursor(),
            publication,
        })
    }
}

enum Run<A: Stores> {
    Local(Simulation<A>),
    #[cfg(not(target_arch = "wasm32"))]
    Thread(SimulationThread<A>),
}

struct Outgoing<P> {
    returned: Option<Publication<P>>,
    restart: Option<Instant>,
}

fn aspect(input: &InputMap, size: (u32, u32)) -> f32 {
    input
        .display_aspect()
        .unwrap_or(size.0 as f32 / size.1 as f32)
}

impl<P> Outgoing<P> {
    fn request(&mut self, input: &mut InputMap, now: Instant, size: (u32, u32)) -> Request<P> {
        Request {
            now,
            aspect: aspect(input, size),
            input: input.take(),
            returned: self.returned.take(),
            restart: self.restart.take(),
        }
    }
}

pub(crate) struct Simulator<A: Stores> {
    run: Run<A>,
    outgoing: Outgoing<A::Published>,
}

impl<A: Stores> Simulator<A> {
    pub(crate) fn local(simulation: Simulation<A>) -> Self {
        Self::new(Run::Local(simulation))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn spawn(simulation: Simulation<A>) -> Result<Self, HostError> {
        Ok(Self::new(Run::Thread(SimulationThread::spawn(simulation)?)))
    }

    fn new(run: Run<A>) -> Self {
        Self {
            run,
            outgoing: Outgoing {
                returned: None,
                restart: None,
            },
        }
    }

    pub(crate) fn session(&self) -> Option<&Session<A>> {
        match &self.run {
            Run::Local(simulation) => Some(&simulation.session),
            #[cfg(not(target_arch = "wasm32"))]
            Run::Thread(_) => None,
        }
    }

    pub(crate) fn reset_clock(&mut self, now: Instant) {
        self.outgoing.restart = Some(now);
    }

    pub(crate) fn release(&mut self, publication: Publication<A::Published>) {
        self.outgoing.returned = Some(publication);
    }

    #[cfg(test)]
    pub(crate) fn released(&self) -> Option<&Publication<A::Published>> {
        self.outgoing.returned.as_ref()
    }

    // On a thread the reply answers the request sent at the end of the last frame, and `before` never runs.
    pub(crate) fn exchange(
        &mut self,
        input: &mut InputMap,
        now: Instant,
        size: (u32, u32),
        before: impl FnOnce(&Session<A>, &Input),
    ) -> Result<Reply<A::Published>, HostError> {
        let Simulator { run, outgoing } = self;
        match run {
            Run::Local(simulation) => {
                let reply = simulation.prepare(outgoing.request(input, now, size), before);
                reply
                    .spent
                    .record(["dispatch", "simulation", "publication"]);
                Ok(reply)
            }
            #[cfg(not(target_arch = "wasm32"))]
            Run::Thread(thread) => {
                if !thread.in_flight {
                    thread.send(outgoing.request(input, now, size))?;
                }
                let reply = {
                    let _wait = frame_trace::scope("simulation wait");
                    thread.receive()?
                };
                reply
                    .spent
                    .record(["thread dispatch", "thread simulation", "thread publication"]);
                Ok(reply)
            }
        }
    }

    // A thread starts the next frame's simulation while this frame draws.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn send_ahead(
        &mut self,
        input: &mut InputMap,
        now: Instant,
        size: (u32, u32),
    ) -> Result<(), HostError> {
        let Simulator { run, outgoing } = self;
        match run {
            Run::Local(_) => Ok(()),
            Run::Thread(thread) => thread.send(outgoing.request(input, now, size)),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct SimulationThread<A: Stores> {
    requests: Option<Sender<Request<A::Published>>>,
    replies: Receiver<Reply<A::Published>>,
    worker: Option<JoinHandle<()>>,
    in_flight: bool,
}

#[cfg(not(target_arch = "wasm32"))]
impl<A: Stores> SimulationThread<A> {
    fn spawn(mut simulation: Simulation<A>) -> Result<Self, HostError> {
        let (requests, inbound) = mpsc::channel::<Request<A::Published>>();
        let (outbound, replies) = mpsc::channel();
        let worker = thread::Builder::new()
            .name("loam-simulation".into())
            .spawn(move || {
                for request in inbound {
                    let reply = simulation.prepare(request, |_, _| {});
                    let fatal = matches!(reply.outcome, Err(Failure::Fatal(_)));
                    if outbound.send(reply).is_err() || fatal {
                        break;
                    }
                }
            })
            .map_err(failed)?;
        Ok(Self {
            requests: Some(requests),
            replies,
            worker: Some(worker),
            in_flight: false,
        })
    }

    fn send(&mut self, request: Request<A::Published>) -> Result<(), HostError> {
        self.requests
            .as_ref()
            .expect("the request channel stays open until the thread is dropped")
            .send(request)
            .map_err(|_| failed("the simulation thread stopped"))?;
        self.in_flight = true;
        Ok(())
    }

    fn receive(&mut self) -> Result<Reply<A::Published>, HostError> {
        self.in_flight = false;
        self.replies
            .recv()
            .map_err(|_| failed("the simulation thread stopped"))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<A: Stores> Drop for SimulationThread<A> {
    fn drop(&mut self) {
        self.requests = None;
        if let Some(worker) = self.worker.take() {
            if worker.join().is_err() {
                tracing::error!("the simulation thread panicked");
            }
        }
    }
}
