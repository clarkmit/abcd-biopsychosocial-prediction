# API reference

The public API is small: build a `PipelineConfig`, run a `Pipeline`, inspect `ctx`.

## Pipeline

::: bpsp.pipeline.Pipeline

::: bpsp.pipeline.working_directory

## Configuration

::: bpsp.config.PipelineConfig

## Run context

`bpsp.ctx` is the single `RunContext` instance shared by all stages.

::: bpsp.context.RunContext

::: bpsp.context.UnsetContextName

## Simulated example data

::: bpsp.simulation
    options:
      members: [make_example_panel, PLANTED_FEATURES]

## Logging

`bpsp.configure_logging`:

::: bpsp._logging.configure_logging
    options:
      show_root_heading: true
      heading_level: 3
