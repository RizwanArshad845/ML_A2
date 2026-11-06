"""Task 3.6: register models in the MLflow Model Registry and set the `champion` alias.

Each run name given becomes a new version of the registered model, in the order given.
The alias `champion` is then pointed at the version made from --champion.

    python src/register_model.py --runs vg_baseline_cnn vg_ref --champion vg_ref
"""
import argparse

import mlflow

import tracking

MODEL_NAME = "visionguard"


def run_id_for(name: str, experiment: str) -> str:
    runs = mlflow.search_runs(experiment_names=[experiment], output_format="list",
                              filter_string=f"attributes.run_name = '{name}'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        raise SystemExit(f"No run named {name!r} in experiment {experiment!r}")
    return runs[0].info.run_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run names to register, in order")
    ap.add_argument("--champion", required=True, help="run name that gets the champion alias")
    ap.add_argument("--experiment", default=tracking.EXPERIMENT)
    ap.add_argument("--name", default=MODEL_NAME)
    args = ap.parse_args()

    tracking.setup(args.experiment)
    client = mlflow.MlflowClient()
    versions = {}
    for run_name in args.runs:
        run_id = run_id_for(run_name, args.experiment)
        mv = mlflow.register_model(f"runs:/{run_id}/model", args.name,
                                   tags={"run_name": run_name})
        versions[run_name] = mv.version
        print(f"registered {args.name} version {mv.version} from run {run_name} ({run_id})")

    if args.champion not in versions:
        raise SystemExit("--champion must be one of --runs")
    client.set_registered_model_alias(args.name, "champion", versions[args.champion])
    print(f"alias 'champion' -> {args.name} version {versions[args.champion]} ({args.champion})")


if __name__ == "__main__":
    main()
