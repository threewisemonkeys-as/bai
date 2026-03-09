import hydra
from omegaconf import DictConfig

from rollout import one_step_wrap
from run_utils import setup_run


@hydra.main(config_path="BALROG/balrog/config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    run_name_suffix = f"{config.agent.type}_{config.client.model_id.replace('/', '_')}_eval"

    original_cwd, output_dir = setup_run(
        config,
        run_name_suffix=run_name_suffix,
        resume_from=config.eval.resume_from,
        output_dir_base=config.eval.output_dir,
    )

    one_step_wrap(config=config, original_cwd=original_cwd, output_dir=output_dir)


if __name__ == "__main__":
    main()
