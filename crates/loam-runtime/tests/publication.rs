use loam_runtime::{DomainError, Phase, PhaseError, Publication, Session, SimConfig};

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Scored {
        frames: Value<u32>,
    }
    publishes Vec<u32>;
}

#[test]
fn the_published_value_is_what_the_latest_value_system_wrote_after_the_publication_systems() {
    let mut session = Session::new(Scored::default(), SimConfig::default());
    session.system(Phase::Publication, "count", |ctx| {
        *ctx.app.frames.get_mut() += 1;
        Ok(())
    });
    session.publish_value("replaced", |_, value: &mut Vec<u32>| {
        value.push(u32::MAX);
        Ok(())
    });
    session.publish_value("frames", |ctx, value: &mut Vec<u32>| {
        let frames = *ctx.app.frames.get();
        if frames == 3 {
            return Err(DomainError::Unsupported("injected"));
        }
        value.clear();
        value.push(frames);
        Ok(())
    });
    let mut publication = Publication::default();

    session.publish(&mut publication, 1.0).unwrap();
    assert_eq!(
        publication.value,
        [1],
        "the value missed this publish's publication systems or kept a replaced system's write"
    );
    session.publish(&mut publication, 1.0).unwrap();
    assert_eq!(publication.value, [2]);

    let failure = PhaseError {
        phase: Phase::Publication,
        system: Some("frames"),
        cause: DomainError::Unsupported("injected"),
    };
    assert_eq!(session.publish(&mut publication, 1.0), Err(failure));
    assert_eq!(
        session.phase_error(),
        Some(failure),
        "a failed value system left the session unfaulted"
    );
}
