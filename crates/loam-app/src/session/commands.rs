use std::sync::{Arc, Mutex, MutexGuard};

use loam_runtime::host::HostError;
use loam_runtime::{
    AppCommand, Command, Dispatch, Growth, Input, Outcome, PhaseError, Rejection, RequestId,
    RestoreError, Session, Setting, Stores,
};

pub(crate) const MAX_PENDING_COMMANDS: usize = 1024;
const MAX_REPORTED_COMMANDS: usize = 256;

// Script holds the 1-based script line that queued the command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Report {
    Silent,
    Console,
    Script(usize),
}

struct Queued<A> {
    name: &'static str,
    command: Command<A>,
    report: Report,
}

struct Submitted {
    request: RequestId,
    name: &'static str,
    report: Report,
    recovery: bool,
}

pub(crate) struct Reported {
    pub name: &'static str,
    pub report: Report,
    pub outcome: Result<Outcome, Rejection>,
}

struct State<A> {
    queued: Vec<Queued<A>>,
    submitted: Vec<Submitted>,
    reported: Vec<Reported>,
    dropped_reports: usize,
    in_flight: usize,
    recovery_in_flight: bool,
}

impl<A> Default for State<A> {
    fn default() -> Self {
        Self {
            queued: Vec::with_capacity(MAX_PENDING_COMMANDS),
            submitted: Vec::with_capacity(MAX_PENDING_COMMANDS),
            reported: Vec::with_capacity(MAX_REPORTED_COMMANDS),
            dropped_reports: 0,
            in_flight: 0,
            recovery_in_flight: false,
        }
    }
}

type Shared<A> = Arc<Mutex<State<A>>>;

fn lock<A>(shared: &Shared<A>) -> MutexGuard<'_, State<A>> {
    shared.lock().unwrap_or_else(|error| error.into_inner())
}

pub struct CommandSender<A: Stores> {
    shared: Shared<A>,
    report: Report,
}

impl<A: Stores> Clone for CommandSender<A> {
    fn clone(&self) -> Self {
        Self {
            shared: self.shared.clone(),
            report: self.report,
        }
    }
}

impl<A: Stores> CommandSender<A> {
    pub fn submit(&self, command: Command<A>) {
        let name = command.name();
        let recovery = matches!(&command, Command::Reset);
        let mut shared = lock(&self.shared);
        let pending = shared.queued.len() + shared.submitted.len() + shared.in_flight;
        let rejected = if recovery {
            pending >= MAX_PENDING_COMMANDS
                || shared.recovery_in_flight
                || shared
                    .queued
                    .iter()
                    .any(|queued| matches!(&queued.command, Command::Reset))
                || shared.submitted.iter().any(|submitted| submitted.recovery)
        } else {
            pending >= MAX_PENDING_COMMANDS - 1
        };
        if rejected {
            deliver(&mut shared, name, self.report, Err(Rejection::Capacity));
            return;
        }
        shared.queued.push(Queued {
            name,
            command,
            report: self.report,
        });
    }

    pub fn app(&self, command: impl AppCommand<A>) {
        self.submit(Command::App(Box::new(command)));
    }

    pub fn app_fn(
        &self,
        name: &'static str,
        mut apply: impl FnMut(&mut Dispatch<'_, A>) + Send + 'static,
    ) {
        self.submit(Command::try_app_fn(name, move |dispatch| {
            apply(dispatch);
            Ok(Outcome::Done)
        }));
    }

    pub fn try_app_fn(
        &self,
        name: &'static str,
        apply: impl FnMut(&mut Dispatch<'_, A>) -> Result<Outcome, Rejection> + Send + 'static,
    ) {
        self.submit(Command::try_app_fn(name, apply));
    }

    pub fn toggle(&self, name: &'static str, setting: Setting<A>) {
        self.submit(Command::toggle(name, setting));
    }

    pub fn set(&self, name: &'static str, setting: Setting<A>, on: bool) {
        self.submit(Command::set(name, setting, on));
    }

    pub fn reset(&self) {
        self.submit(Command::Reset);
    }

    pub(crate) fn reported(&self) -> Self {
        Self {
            shared: self.shared.clone(),
            report: Report::Console,
        }
    }

    pub(crate) fn set_report(&mut self, report: Report) {
        self.report = report;
    }

    pub(crate) fn take_reported(&self, into: &mut Vec<Reported>) -> usize {
        let mut shared = lock(&self.shared);
        into.append(&mut shared.reported);
        std::mem::take(&mut shared.dropped_reports)
    }
}

fn deliver<A>(
    shared: &mut State<A>,
    name: &'static str,
    report: Report,
    outcome: Result<Outcome, Rejection>,
) {
    if report == Report::Silent {
        match &outcome {
            Ok(_) => return,
            Err(rejection) => tracing::warn!("{name}: rejected, {rejection}"),
        }
    }
    if shared.reported.len() < MAX_REPORTED_COMMANDS {
        shared.reported.push(Reported {
            name,
            report,
            outcome,
        });
    } else {
        shared.dropped_reports = shared.dropped_reports.saturating_add(1);
    }
}

pub(crate) struct CommandInbox<A: Stores> {
    shared: Shared<A>,
}

impl<A: Stores> Clone for CommandInbox<A> {
    fn clone(&self) -> Self {
        Self {
            shared: self.shared.clone(),
        }
    }
}

impl<A: Stores> Default for CommandInbox<A> {
    fn default() -> Self {
        Self {
            shared: Shared::default(),
        }
    }
}

impl<A: Stores> CommandInbox<A> {
    pub fn sender(&self) -> CommandSender<A> {
        CommandSender {
            shared: self.shared.clone(),
            report: Report::Silent,
        }
    }

    // The first value is the fault a queued reset cleared.
    pub fn boundary(
        &self,
        session: &mut Session<A>,
        input: Input,
    ) -> (Option<PhaseError>, Result<Growth, HostError>) {
        let fault = session.faulted_phase().and_then(|_| session.phase_error());
        let recovery = self.recover(session);
        let recovered = fault.filter(|_| recovery.is_ok() && session.faulted_phase().is_none());
        let result = recovery.map_err(HostError::from).and_then(|()| {
            self.drain(session);
            session.boundary(input).map_err(HostError::from)
        });
        self.collect(session);
        (recovered, result)
    }

    pub fn recover(&self, session: &mut Session<A>) -> Result<(), RestoreError> {
        if session.faulted_phase().is_none() {
            return Ok(());
        }
        let (mut batch, reset) = {
            let mut shared = lock(&self.shared);
            let Some(reset) = shared
                .queued
                .iter()
                .position(|queued| matches!(&queued.command, Command::Reset))
            else {
                return Ok(());
            };
            let batch =
                std::mem::replace(&mut shared.queued, Vec::with_capacity(MAX_PENDING_COMMANDS));
            shared.in_flight += batch.len();
            shared.recovery_in_flight = true;
            (batch, reset)
        };
        let outcome = session.reset();
        self.collect(session);
        let mut shared = lock(&self.shared);
        shared.in_flight -= batch.len();
        shared.recovery_in_flight = false;
        for (index, queued) in batch.drain(..).enumerate() {
            let result = if index == reset {
                outcome.map(|()| Outcome::Done).map_err(Rejection::Restore)
            } else {
                Err(Rejection::Cancelled)
            };
            deliver(&mut shared, queued.name, queued.report, result);
        }
        outcome
    }

    pub fn drain(&self, session: &mut Session<A>) {
        let mut shared = lock(&self.shared);
        let State {
            queued, submitted, ..
        } = &mut *shared;
        for queued in queued.drain(..) {
            let recovery = matches!(&queued.command, Command::Reset);
            submitted.push(Submitted {
                request: session.submit(queued.command),
                name: queued.name,
                report: queued.report,
                recovery,
            });
        }
    }

    pub fn collect(&self, session: &Session<A>) {
        let mut shared = lock(&self.shared);
        for result in session.results() {
            let Some(index) = shared
                .submitted
                .iter()
                .position(|submitted| submitted.request == result.request)
            else {
                if result.outcome.is_err() {
                    deliver(&mut shared, result.name, Report::Silent, result.outcome);
                }
                continue;
            };
            let submitted = shared.submitted.swap_remove(index);
            deliver(
                &mut shared,
                submitted.name,
                submitted.report,
                result.outcome,
            );
        }
    }
}
