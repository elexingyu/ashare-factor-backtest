"""Versioned production-job contract and machine inspection service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import resource
import sys
from typing import Any, Mapping

import pandas as pd
import yaml

from ashare_factor_backtest.application.inspect_plugin import PluginInspectionService
from ashare_factor_backtest.data.manifest import content_fingerprint
from ashare_factor_backtest.expression.parser import referenced_fields
from ashare_factor_backtest.evaluation.production_chunked_evaluator import (
    evaluate_expression_by_year,
)
from ashare_factor_backtest.evaluation.production_context import INDEX_MEMBERSHIP_VIEWS
from ashare_factor_backtest.evaluation.production_frame_loader import (
    BatchedProductionFrameLoader,
)
from ashare_factor_backtest.plugins.production_bridge import ProductionPluginFieldSource
from ashare_factor_backtest.evaluation.production_rolling import (
    ProductionRollingGate,
    ProductionRollingPolicy,
    ProductionRollingWindow,
)
from ashare_factor_backtest.evaluation.production_screen import ProductionScreenPolicy
from ashare_factor_backtest.evaluation.policies import ProductionSelectionNullPolicy
from ashare_factor_backtest.evaluation.production_universe_readiness import YearChunk


_PLAIN_VIEWS = frozenset({"signal_eligible", "mainboard", "liquid_20m"})
_INDEX_VIEWS = dict(INDEX_MEMBERSHIP_VIEWS)


@dataclass(frozen=True)
class ProductionDataPaths:
    bars_dir: Path
    market_states_dir: Path
    stock_master: Path
    st_intervals: Path
    trade_calendar: Path
    index_membership: Path | None


@dataclass(frozen=True)
class ProductionPluginBinding:
    manifest: Path
    fields: tuple[str, ...]


@dataclass(frozen=True)
class ProductionEvaluationSpec:
    start: date
    end: date
    max_lookback: int
    symbol_batch_size: int
    cache_mib: int
    memory_limit_mib: float
    symbol_cap: int | None


@dataclass(frozen=True)
class ProductionResearchSpec:
    evidence_mode: str
    screen: ProductionScreenPolicy
    rolling: ProductionRollingPolicy
    rolling_gate: ProductionRollingGate
    selection_null: ProductionSelectionNullPolicy
    selection_null_seed_namespace: str

    def __post_init__(self) -> None:
        if self.evidence_mode not in {"engineering", "production"}:
            raise ValueError(
                "production research evidence_mode must be engineering or production"
            )
        if not self.selection_null_seed_namespace.strip():
            raise ValueError(
                "production research selection-null seed_namespace must be nonempty"
            )
        minimum_p = 1.0 / (self.selection_null.permutations + 1)
        if (
            self.evidence_mode == "production"
            and minimum_p > self.selection_null.maximum_empirical_p
        ):
            raise ValueError(
                "production selection-null p-value resolution cannot reach its gate"
            )


@dataclass(frozen=True)
class ProductionJob:
    source_path: Path
    job_id: str
    dataset_version: str
    data: ProductionDataPaths
    plugins: tuple[ProductionPluginBinding, ...]
    view: str
    index_code: str | None
    evaluation: ProductionEvaluationSpec
    research: ProductionResearchSpec | None = None

    @property
    def contract_identity(self) -> str:
        payload = {
            "data": {
                "bars_dir": str(self.data.bars_dir),
                "index_membership": (
                    str(self.data.index_membership)
                    if self.data.index_membership is not None
                    else None
                ),
                "market_states_dir": str(self.data.market_states_dir),
                "st_intervals": str(self.data.st_intervals),
                "stock_master": str(self.data.stock_master),
                "trade_calendar": str(self.data.trade_calendar),
            },
            "dataset_version": self.dataset_version,
            "evaluation": {
                "cache_mib": self.evaluation.cache_mib,
                "end": self.evaluation.end.isoformat(),
                "max_lookback": self.evaluation.max_lookback,
                "memory_limit_mib": self.evaluation.memory_limit_mib,
                "start": self.evaluation.start.isoformat(),
                "symbol_batch_size": self.evaluation.symbol_batch_size,
                "symbol_cap": self.evaluation.symbol_cap,
            },
            "job_id": self.job_id,
            "plugins": [
                {"fields": list(item.fields), "manifest": str(item.manifest)}
                for item in self.plugins
            ],
            "schema_version": "production-job.v1",
            "universe": {"index_code": self.index_code, "view": self.view},
        }
        if self.research is not None:
            payload["research"] = production_research_contract(self.research)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PreparedProductionJob:
    job: ProductionJob
    inspected: dict[str, object]
    warnings: tuple[str, ...]
    symbols: tuple[str, ...]
    open_dates: pd.DatetimeIndex
    calculation_dates: pd.DatetimeIndex
    chunks: tuple[YearChunk, ...]
    frame_loader: BatchedProductionFrameLoader

    @property
    def job_identity(self) -> str:
        return str(self.inspected["job_identity"])


def load_production_job(path: Path) -> ProductionJob:
    source = Path(path).resolve(strict=True)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read production job: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("production job must contain an object")
    if payload.get("schema_version") != "production-job.v1":
        raise ValueError("unsupported production job schema_version")
    root = source.parent
    data = _mapping(payload, "data")
    universe = _mapping(payload, "universe")
    evaluation = _mapping(payload, "evaluation")
    job_id = _text(payload, "job_id")
    dataset_version = _text(payload, "dataset_version")
    view = _text(universe, "view")
    index_code_raw = universe.get("index_code")
    index_code = None if index_code_raw is None else str(index_code_raw).strip()
    expected_index = _INDEX_VIEWS.get(view)
    if view in _PLAIN_VIEWS:
        if index_code is not None:
            raise ValueError(f"production view {view} must not declare index_code")
    elif expected_index is None:
        raise ValueError(f"unsupported production job universe view: {view}")
    elif index_code != expected_index:
        raise ValueError(f"production view {view} requires index_code {expected_index}")

    index_value = data.get("index_membership")
    paths = ProductionDataPaths(
        bars_dir=_path(root, data, "bars_dir"),
        market_states_dir=_path(root, data, "market_states_dir"),
        stock_master=_path(root, data, "stock_master"),
        st_intervals=_path(root, data, "st_intervals"),
        trade_calendar=_path(root, data, "trade_calendar"),
        index_membership=(
            None
            if index_value is None
            else (root / str(index_value)).resolve(strict=False)
        ),
    )
    if expected_index is not None and paths.index_membership is None:
        raise ValueError(f"production view {view} requires index_membership")

    plugins_payload = payload.get("plugins", [])
    if not isinstance(plugins_payload, list):
        raise ValueError("production job plugins must be a list")
    plugins: list[ProductionPluginBinding] = []
    owned_fields: set[str] = set()
    for raw in plugins_payload:
        if not isinstance(raw, Mapping):
            raise ValueError("production job plugins must contain objects")
        fields_raw = raw.get("fields")
        if not isinstance(fields_raw, list) or not fields_raw:
            raise ValueError("production plugin fields must be a non-empty list")
        fields = tuple(str(value) for value in fields_raw)
        if len(set(fields)) != len(fields):
            raise ValueError("production plugin fields must be unique")
        overlap = owned_fields.intersection(fields)
        if overlap:
            raise ValueError(
                f"production plugin fields have multiple owners: {sorted(overlap)}"
            )
        owned_fields.update(fields)
        plugins.append(
            ProductionPluginBinding(
                manifest=_path(root, raw, "manifest"),
                fields=fields,
            )
        )

    start = _date(evaluation, "start")
    end = _date(evaluation, "end")
    if start > end:
        raise ValueError("production job start must not exceed end")
    max_lookback = _positive_int(evaluation, "max_lookback")
    symbol_batch_size = _positive_int(evaluation, "symbol_batch_size")
    cache_mib = _positive_int(evaluation, "cache_mib")
    memory_limit = evaluation.get("memory_limit_mib")
    if (
        isinstance(memory_limit, bool)
        or not isinstance(memory_limit, (int, float))
        or memory_limit <= 0
    ):
        raise ValueError("production job memory_limit_mib must be positive")
    symbol_cap_raw = evaluation.get("symbol_cap")
    symbol_cap = (
        None if symbol_cap_raw is None else _positive_int(evaluation, "symbol_cap")
    )
    evaluation_spec = ProductionEvaluationSpec(
        start=start,
        end=end,
        max_lookback=max_lookback,
        symbol_batch_size=symbol_batch_size,
        cache_mib=cache_mib,
        memory_limit_mib=float(memory_limit),
        symbol_cap=symbol_cap,
    )
    research = _parse_research(payload.get("research"), evaluation_spec)
    if (
        research is not None
        and research.evidence_mode == "production"
        and symbol_cap is not None
    ):
        raise ValueError("production research mode must not declare symbol_cap")
    return ProductionJob(
        source_path=source,
        job_id=job_id,
        dataset_version=dataset_version,
        data=paths,
        plugins=tuple(plugins),
        view=view,
        index_code=index_code,
        evaluation=evaluation_spec,
        research=research,
    )


def _parse_research(
    value: Any,
    evaluation: ProductionEvaluationSpec,
) -> ProductionResearchSpec | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("production job research must be an object")
    evidence_mode = _text(value, "evidence_mode")
    screen_raw = _mapping(value, "screen")
    discovery = _date_pair(screen_raw, "discovery")
    validation = _date_pair(screen_raw, "validation")
    if date.fromisoformat(discovery[1]) >= date.fromisoformat(validation[0]):
        raise ValueError(
            "production screen discovery must end before validation starts"
        )
    horizons = _positive_int_tuple(screen_raw, "horizons")
    minimum_coverage = _unit_interval(
        screen_raw, "minimum_coverage", include_zero=False
    )
    minimum_periods = _positive_int(screen_raw, "minimum_periods")
    top_fraction = _unit_interval(screen_raw, "top_fraction", include_zero=False)
    costs = {
        name: _cost(screen_raw, name)
        for name in (
            "real_buy_cost",
            "real_sell_cost",
            "stress_buy_cost",
            "stress_sell_cost",
        )
    }
    screen = ProductionScreenPolicy(
        discovery=discovery,
        validation=validation,
        horizons=horizons,
        minimum_coverage=minimum_coverage,
        minimum_periods=minimum_periods,
        top_fraction=top_fraction,
        **costs,
    )

    rolling_raw = _mapping(value, "rolling")
    windows_raw = rolling_raw.get("windows")
    if not isinstance(windows_raw, list) or not windows_raw:
        raise ValueError("production research rolling windows must be a non-empty list")
    windows: list[ProductionRollingWindow] = []
    for raw in windows_raw:
        if not isinstance(raw, Mapping):
            raise ValueError("production research rolling windows must contain objects")
        train = _date_pair(raw, "train")
        test = _date_pair(raw, "test")
        windows.append(
            ProductionRollingWindow(
                label=_text(raw, "label"),
                train_start=train[0],
                train_end=train[1],
                test_start=test[0],
                test_end=test[1],
            )
        )
    rolling = ProductionRollingPolicy(
        windows=tuple(windows),
        horizons=screen.horizons,
        minimum_coverage=screen.minimum_coverage,
        minimum_periods=screen.minimum_periods,
        top_fraction=screen.top_fraction,
        real_buy_cost=screen.real_buy_cost,
        real_sell_cost=screen.real_sell_cost,
        stress_buy_cost=screen.stress_buy_cost,
        stress_sell_cost=screen.stress_sell_cost,
    )
    gate_raw = _mapping(rolling_raw, "gate")
    rolling_gate = ProductionRollingGate(
        required_folds=_positive_int(gate_raw, "required_folds"),
        minimum_positive_folds=_positive_int(gate_raw, "minimum_positive_folds"),
        minimum_median_test_excess_sharpe=_finite_number(
            gate_raw, "minimum_median_test_excess_sharpe"
        ),
        minimum_direction_mode_count=_positive_int(
            gate_raw, "minimum_direction_mode_count"
        ),
        minimum_horizon_mode_count=_positive_int(
            gate_raw, "minimum_horizon_mode_count"
        ),
    )
    if rolling_gate.required_folds != len(windows):
        raise ValueError("production rolling required_folds must equal window count")
    for count in (
        rolling_gate.minimum_positive_folds,
        rolling_gate.minimum_direction_mode_count,
        rolling_gate.minimum_horizon_mode_count,
    ):
        if count > rolling_gate.required_folds:
            raise ValueError("production rolling gate count exceeds required_folds")

    null_raw = _mapping(value, "selection_null")
    selection_null = ProductionSelectionNullPolicy(
        permutations=_positive_int(null_raw, "permutations"),
        maximum_portfolio_evaluations=_positive_int(
            null_raw, "maximum_portfolio_evaluations"
        ),
        maximum_empirical_p=_unit_interval(
            null_raw, "maximum_empirical_p", include_zero=False
        ),
        minimum_changed_column_fraction=_unit_interval(
            null_raw, "minimum_changed_column_fraction", include_zero=True
        ),
        maximum_mean_coverage_drift=_unit_interval(
            null_raw, "maximum_mean_coverage_drift", include_zero=True
        ),
        maximum_daily_coverage_drift=_unit_interval(
            null_raw, "maximum_daily_coverage_drift", include_zero=True
        ),
    )
    research = ProductionResearchSpec(
        evidence_mode=evidence_mode,
        screen=screen,
        rolling=rolling,
        rolling_gate=rolling_gate,
        selection_null=selection_null,
        selection_null_seed_namespace=_text(null_raw, "seed_namespace"),
    )
    _validate_research_dates(research, evaluation)
    return research


def _validate_research_dates(
    research: ProductionResearchSpec,
    evaluation: ProductionEvaluationSpec,
) -> None:
    lower = evaluation.start
    upper = evaluation.end
    ranges = [research.screen.discovery, research.screen.validation]
    ranges.extend(
        (start, end)
        for window in research.rolling.windows
        for start, end in (
            (window.train_start, window.train_end),
            (window.test_start, window.test_end),
        )
    )
    if any(
        date.fromisoformat(start) < lower or date.fromisoformat(end) > upper
        for start, end in ranges
    ):
        raise ValueError("production research dates exceed the evaluation range")


def production_research_contract(
    research: ProductionResearchSpec,
) -> dict[str, object]:
    """Return the stable, serializable research contract used by a job."""
    screen = research.screen
    gate = research.rolling_gate
    null = research.selection_null
    return {
        "evidence_mode": research.evidence_mode,
        "rolling": {
            "gate": {
                "minimum_direction_mode_count": gate.minimum_direction_mode_count,
                "minimum_horizon_mode_count": gate.minimum_horizon_mode_count,
                "minimum_median_test_excess_sharpe": (
                    gate.minimum_median_test_excess_sharpe
                ),
                "minimum_positive_folds": gate.minimum_positive_folds,
                "required_folds": gate.required_folds,
            },
            "windows": [
                {
                    "label": window.label,
                    "test": [window.test_start, window.test_end],
                    "train": [window.train_start, window.train_end],
                }
                for window in research.rolling.windows
            ],
        },
        "screen": {
            "discovery": list(screen.discovery),
            "horizons": list(screen.horizons),
            "minimum_coverage": screen.minimum_coverage,
            "minimum_periods": screen.minimum_periods,
            "real_buy_cost": screen.real_buy_cost,
            "real_sell_cost": screen.real_sell_cost,
            "stress_buy_cost": screen.stress_buy_cost,
            "stress_sell_cost": screen.stress_sell_cost,
            "top_fraction": screen.top_fraction,
            "validation": list(screen.validation),
        },
        "selection_null": {
            "maximum_daily_coverage_drift": null.maximum_daily_coverage_drift,
            "maximum_empirical_p": null.maximum_empirical_p,
            "maximum_mean_coverage_drift": null.maximum_mean_coverage_drift,
            "maximum_portfolio_evaluations": null.maximum_portfolio_evaluations,
            "minimum_changed_column_fraction": (null.minimum_changed_column_fraction),
            "permutations": null.permutations,
            "seed_namespace": research.selection_null_seed_namespace,
        },
    }


class ProductionJobService:
    def prepare(
        self,
        path: Path,
        *,
        expression: str,
        validation_cache_root: Path | None = None,
    ) -> PreparedProductionJob:
        return self._prepare(
            path,
            requested=set(referenced_fields(expression)),
            validation_cache_root=validation_cache_root,
        )

    def prepare_batch(
        self,
        path: Path,
        *,
        expressions: tuple[str, ...],
        validation_cache_root: Path | None = None,
    ) -> PreparedProductionJob:
        if not expressions or any(
            not str(expression).strip() for expression in expressions
        ):
            raise ValueError("production job batch expressions must be non-empty")
        requested = set().union(
            *(set(referenced_fields(expression)) for expression in expressions)
        )
        return self._prepare(
            path,
            requested=requested,
            validation_cache_root=validation_cache_root,
        )

    def prepare_fields(
        self,
        path: Path,
        *,
        requested_fields: set[str],
        validation_cache_root: Path | None = None,
    ) -> PreparedProductionJob:
        requested = {str(field).strip() for field in requested_fields}
        if not requested or any(not field for field in requested):
            raise ValueError("production job requested fields must be non-empty")
        return self._prepare(
            path,
            requested=requested,
            validation_cache_root=validation_cache_root,
        )

    def _prepare(
        self,
        path: Path,
        *,
        requested: set[str],
        validation_cache_root: Path | None,
    ) -> PreparedProductionJob:
        job = load_production_job(path)
        plugin_source = (
            ProductionPluginFieldSource(
                tuple(binding.manifest for binding in job.plugins),
                validation_cache_root=validation_cache_root,
            )
            if job.plugins
            else None
        )
        inspected, inspect_warnings = self.inspect(
            path, plugin_field_source=plugin_source
        )
        selected_plugin_fields = {
            field
            for binding in job.plugins
            for field in requested.intersection(binding.fields)
        }
        symbols, index_views = _resolve_job_symbols(job)
        warnings = list(inspect_warnings)
        if job.evaluation.symbol_cap is not None:
            symbols = symbols[: job.evaluation.symbol_cap]
            warnings.append(
                "Engineering symbol cap is active; results are non-promotable."
            )
        if not symbols:
            raise ValueError("production job symbol selection is empty")
        open_dates, calculation_dates = _job_trading_dates(job)
        bars_manifest = _json_object(
            job.data.bars_dir / "manifest.json", "production bars manifest"
        )
        analysis_range = bars_manifest["analysis_range"]
        chunks = _bounded_job_chunks(
            open_dates,
            calculation_dates,
            max_lookback=job.evaluation.max_lookback,
            available_start=pd.Timestamp(str(analysis_range[0])),
        )
        loader = BatchedProductionFrameLoader(
            bars_dir=job.data.bars_dir,
            market_states_dir=job.data.market_states_dir,
            stock_master_path=job.data.stock_master,
            st_intervals_path=job.data.st_intervals,
            trade_calendar_path=job.data.trade_calendar,
            batch_size=job.evaluation.symbol_batch_size,
            plugin_field_source=plugin_source,
            plugin_fields=selected_plugin_fields,
            index_membership_path=(job.data.index_membership if index_views else None),
            index_membership_views=index_views,
            symbols=symbols,
        )
        return PreparedProductionJob(
            job=job,
            inspected=inspected,
            warnings=tuple(warnings),
            symbols=symbols,
            open_dates=open_dates,
            calculation_dates=calculation_dates,
            chunks=chunks,
            frame_loader=loader,
        )

    def inspect(
        self,
        path: Path,
        *,
        plugin_field_source: ProductionPluginFieldSource | None = None,
    ) -> tuple[dict[str, object], tuple[str, ...]]:
        job = load_production_job(path)
        bars_manifest_path = job.data.bars_dir / "manifest.json"
        bars_manifest = _json_object(bars_manifest_path, "production bars manifest")
        if bars_manifest.get("schema") != "astock_production_yearly_bars_v1":
            raise ValueError("production bars manifest schema is invalid")
        bars_identity = str(bars_manifest.get("content_hash", ""))
        if len(bars_identity) != 64:
            raise ValueError("production bars manifest content_hash is invalid")
        analysis_range = bars_manifest.get("analysis_range")
        if not isinstance(analysis_range, list) or len(analysis_range) != 2:
            raise ValueError("production bars manifest analysis_range is invalid")
        if (
            date.fromisoformat(str(analysis_range[0])) > job.evaluation.start
            or date.fromisoformat(str(analysis_range[1])) < job.evaluation.end
        ):
            raise ValueError("production bars do not cover the job evaluation range")

        for value, label, directory in (
            (job.data.market_states_dir, "market states", True),
            (job.data.stock_master, "stock master", False),
            (job.data.st_intervals, "ST intervals", False),
            (job.data.trade_calendar, "trade calendar", False),
        ):
            if directory and not value.is_dir():
                raise ValueError(
                    f"production job {label} directory is missing: {value}"
                )
            if not directory and not value.is_file():
                raise ValueError(f"production job {label} file is missing: {value}")
        state_parts = tuple(
            sorted(job.data.market_states_dir.glob("year=*/part.parquet"))
        )
        if not state_parts:
            raise ValueError(
                "production job market states contain no yearly partitions"
            )
        state_manifest_path = job.data.market_states_dir / "manifest.json"
        warnings: list[str] = []
        if job.research is not None:
            minimum_p = 1.0 / (job.research.selection_null.permutations + 1)
            if minimum_p > job.research.selection_null.maximum_empirical_p:
                warnings.append(
                    "Engineering selection-null permutations cannot reach the configured "
                    f"p-value gate; minimum attainable p is {minimum_p:.6f}."
                )
        if state_manifest_path.is_file():
            state_manifest = _json_object(state_manifest_path, "market states manifest")
            states_identity = str(state_manifest.get("content_hash", ""))
            if len(states_identity) != 64:
                raise ValueError("market states manifest content_hash is invalid")
        else:
            states_identity = content_fingerprint(state_parts)
            warnings.append(
                "Market states have no manifest; identity was computed from partition bytes."
            )

        asset_identities = {
            "bars": bars_identity,
            "market_states": states_identity,
            "st_intervals": content_fingerprint((job.data.st_intervals,)),
            "stock_master": content_fingerprint((job.data.stock_master,)),
            "trade_calendar": content_fingerprint((job.data.trade_calendar,)),
        }
        if job.data.index_membership is not None:
            if not job.data.index_membership.is_file():
                raise ValueError("production job index_membership file is missing")
            manifest = _json_object(
                job.data.index_membership.parent / "manifest.json",
                "index membership manifest",
            )
            identity = str(manifest.get("content_hash", ""))
            if len(identity) != 64:
                raise ValueError("index membership content_hash is invalid")
            asset_identities["index_membership"] = identity

        expected_plugin_paths = tuple(
            binding.manifest.resolve() for binding in job.plugins
        )
        if plugin_field_source is None and expected_plugin_paths:
            plugin_field_source = ProductionPluginFieldSource(expected_plugin_paths)
        if plugin_field_source is not None:
            actual_plugin_paths = tuple(
                sorted(plugin_field_source.manifests_by_path, key=str)
            )
            if tuple(sorted(expected_plugin_paths, key=str)) != actual_plugin_paths:
                raise ValueError(
                    "production job validated plugin source does not match job"
                )
        plugin_rows = []
        for binding in job.plugins:
            assert plugin_field_source is not None
            manifest = plugin_field_source.manifests_by_path[binding.manifest.resolve()]
            data, plugin_warnings = PluginInspectionService().describe(
                manifest, fields=binding.fields
            )
            ineligible = [
                str(field["name"])
                for field in data["fields"]
                if not field["production_next_open_eligible"]
            ]
            if ineligible:
                raise ValueError(
                    f"production job plugin fields require availability transforms: {ineligible}"
                )
            warnings.extend(plugin_warnings)
            plugin_rows.append(
                {
                    "dataset": data["dataset"],
                    "fields": list(binding.fields),
                    "manifest": str(binding.manifest),
                    "manifest_identity": data["manifest_identity"],
                }
            )

        identity_payload = {
            "assets": asset_identities,
            "contract_identity": job.contract_identity,
            "plugins": plugin_rows,
        }
        job_identity = hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        return (
            {
                "asset_identities": asset_identities,
                "contract_identity": job.contract_identity,
                "dataset_version": job.dataset_version,
                "evaluation": {
                    "end": job.evaluation.end.isoformat(),
                    "max_lookback": job.evaluation.max_lookback,
                    "start": job.evaluation.start.isoformat(),
                },
                "job_id": job.job_id,
                "job_identity": job_identity,
                "plugins": plugin_rows,
                "resources": {
                    "cache_mib": job.evaluation.cache_mib,
                    "memory_limit_mib": job.evaluation.memory_limit_mib,
                    "symbol_batch_size": job.evaluation.symbol_batch_size,
                    "symbol_cap": job.evaluation.symbol_cap,
                },
                "universe": {"index_code": job.index_code, "view": job.view},
            },
            tuple(warnings),
        )

    def smoke(
        self, path: Path, expression: str
    ) -> tuple[dict[str, object], tuple[str, ...]]:
        job = load_production_job(path)
        plugin_source = (
            ProductionPluginFieldSource(
                tuple(binding.manifest for binding in job.plugins)
            )
            if job.plugins
            else None
        )
        inspected, inspect_warnings = self.inspect(
            path, plugin_field_source=plugin_source
        )
        requested = set(referenced_fields(expression))
        selected_plugin_fields: set[str] = set()
        for binding in job.plugins:
            used = requested.intersection(binding.fields)
            if used:
                selected_plugin_fields.update(used)

        master_codes = tuple(
            sorted(
                pd.read_parquet(job.data.stock_master, columns=["ts_code"])["ts_code"]
                .dropna()
                .astype(str)
                .unique()
            )
        )
        if not master_codes:
            raise ValueError("production job stock master contains no symbols")
        symbols = master_codes
        index_views: dict[str, str] = {}
        if job.index_code is not None:
            assert job.data.index_membership is not None
            membership_frame = pd.read_parquet(
                job.data.index_membership,
                columns=["index_code", "effective_date", "con_code"],
                filters=[("index_code", "==", job.index_code)],
            )
            if job.evaluation.symbol_cap is None:
                members = set(membership_frame["con_code"].astype(str))
            else:
                effective = pd.to_datetime(
                    membership_frame["effective_date"], errors="coerce"
                )
                eligible = membership_frame.loc[
                    effective.le(pd.Timestamp(job.evaluation.end))
                ].copy()
                if eligible.empty:
                    raise ValueError(
                        "production job index membership has no effective snapshot"
                    )
                latest = pd.to_datetime(eligible["effective_date"]).max()
                members = set(
                    eligible.loc[
                        pd.to_datetime(eligible["effective_date"]).eq(latest),
                        "con_code",
                    ].astype(str)
                )
            symbols = tuple(code for code in master_codes if code in members)
            index_views = {job.view: job.index_code}
            if not symbols:
                raise ValueError(
                    "production job index membership has no master symbols"
                )
        warnings = list(inspect_warnings)
        if job.evaluation.symbol_cap is not None:
            symbols = symbols[: job.evaluation.symbol_cap]
            warnings.append(
                "Symbol cap is active; smoke output is engineering evidence only."
            )
        if not symbols:
            raise ValueError("production job symbol selection is empty")

        calendar = pd.read_csv(job.data.trade_calendar)
        required_calendar = {"exchange", "cal_date", "is_open"}
        missing_calendar = sorted(required_calendar.difference(calendar.columns))
        if missing_calendar:
            raise ValueError(
                f"production trade calendar is missing: {missing_calendar}"
            )
        calendar_dates = pd.to_datetime(
            calendar["cal_date"].astype("string").str.replace(r"\.0$", "", regex=True),
            format="%Y%m%d",
            errors="coerce",
        )
        calendar_coverage = pd.DatetimeIndex(
            calendar_dates.loc[calendar["exchange"].isin(["SSE", "SZSE"])]
        ).dropna()
        if (
            calendar_coverage.empty
            or calendar_coverage.min() > pd.Timestamp(job.evaluation.start)
            or calendar_coverage.max() < pd.Timestamp(job.evaluation.end)
        ):
            raise ValueError("production trade calendar does not cover the job range")
        open_dates = (
            pd.DatetimeIndex(
                calendar_dates.loc[
                    calendar["exchange"].isin(["SSE", "SZSE"])
                    & pd.to_numeric(calendar["is_open"], errors="coerce").eq(1)
                    & calendar_dates.le(pd.Timestamp(job.evaluation.end))
                ]
            )
            .dropna()
            .drop_duplicates()
            .sort_values()
        )
        calculation_dates = open_dates[
            (open_dates >= pd.Timestamp(job.evaluation.start))
            & (open_dates <= pd.Timestamp(job.evaluation.end))
        ]
        if calculation_dates.empty:
            raise ValueError(
                "production job evaluation range has no open trading dates"
            )
        bars_manifest = _json_object(
            job.data.bars_dir / "manifest.json", "production bars manifest"
        )
        analysis_range = bars_manifest["analysis_range"]
        chunks = _bounded_job_chunks(
            open_dates,
            calculation_dates,
            max_lookback=job.evaluation.max_lookback,
            available_start=pd.Timestamp(str(analysis_range[0])),
        )
        loader = BatchedProductionFrameLoader(
            bars_dir=job.data.bars_dir,
            market_states_dir=job.data.market_states_dir,
            stock_master_path=job.data.stock_master,
            st_intervals_path=job.data.st_intervals,
            trade_calendar_path=job.data.trade_calendar,
            batch_size=job.evaluation.symbol_batch_size,
            plugin_field_source=plugin_source,
            plugin_fields=selected_plugin_fields,
            index_membership_path=(job.data.index_membership if index_views else None),
            index_membership_views=index_views,
            symbols=symbols,
        )
        with redirect_stdout(sys.stderr):
            evaluated = evaluate_expression_by_year(
                expression,
                chunks=chunks,
                frame_loader=loader,
                dataset_version=(
                    f"{job.dataset_version}_{str(inspected['job_identity'])[:16]}"
                ),
                view=job.view,
                cache_max_bytes=job.evaluation.cache_mib * 1024 * 1024,
                required_fields=requested,
                spill_to_disk=True,
            )
        finite_cells = int(evaluated.values.notna().sum().sum())
        total_cells = int(evaluated.values.shape[0] * evaluated.values.shape[1])
        if finite_cells == 0:
            raise ValueError("production job smoke produced no finite factor values")
        peak_rss_mib = _peak_rss_mib()
        if peak_rss_mib > job.evaluation.memory_limit_mib:
            raise MemoryError(
                f"production job smoke exceeded memory limit: {peak_rss_mib:.1f} MiB"
            )
        return (
            {
                "canonical": evaluated.canonical,
                "chunk_count": len(evaluated.chunks),
                "chunks": [
                    {
                        "calculation_end": chunk.calculation_end,
                        "calculation_start": chunk.calculation_start,
                        "date_count": chunk.date_count,
                        "finite_cells": chunk.finite_cells,
                        "security_count": chunk.security_count,
                        "year": chunk.year,
                    }
                    for chunk in evaluated.chunks
                ],
                "date_count": len(evaluated.values.index),
                "factor_id": evaluated.factor_id,
                "finite_cells": finite_cells,
                "finite_coverage": finite_cells / total_cells if total_cells else 0.0,
                "job_id": job.job_id,
                "job_identity": inspected["job_identity"],
                "lookback": evaluated.lookback,
                "peak_rss_mib": peak_rss_mib,
                "effective_trading_range": {
                    "end": pd.Timestamp(evaluated.values.index.max())
                    .date()
                    .isoformat(),
                    "start": pd.Timestamp(evaluated.values.index.min())
                    .date()
                    .isoformat(),
                },
                "requested_calendar_range": {
                    "end": job.evaluation.end.isoformat(),
                    "start": job.evaluation.start.isoformat(),
                },
                "return_data_read": False,
                "security_count": len(evaluated.values.columns),
            },
            tuple(warnings),
        )


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"production job {key} must be an object")
    return value


def _text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"production job {key} must be normalized non-empty text")
    return value


def _path(root: Path, payload: Mapping[str, Any], key: str) -> Path:
    value = _text(payload, key)
    return (root / value).resolve(strict=False)


def _date(payload: Mapping[str, Any], key: str) -> date:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"production job {key} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"production job {key} must be an ISO date") from error


def _positive_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"production job {key} must be a positive integer")
    return value


def _positive_int_tuple(payload: Mapping[str, Any], key: str) -> tuple[int, ...]:
    values = payload.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"production job {key} must be a non-empty list")
    parsed = tuple(values)
    if len(set(parsed)) != len(parsed) or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in parsed
    ):
        raise ValueError(f"production job {key} must contain unique positive integers")
    return parsed


def _finite_number(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"production job {key} must be a finite number")
    parsed = float(value)
    if not pd.notna(parsed) or parsed in {float("inf"), float("-inf")}:
        raise ValueError(f"production job {key} must be a finite number")
    return parsed


def _cost(payload: Mapping[str, Any], key: str) -> float:
    value = _finite_number(payload, key)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"production research cost {key} must be in [0, 1)")
    return value


def _unit_interval(
    payload: Mapping[str, Any], key: str, *, include_zero: bool
) -> float:
    value = _finite_number(payload, key)
    valid = 0.0 <= value <= 1.0 if include_zero else 0.0 < value <= 1.0
    if not valid:
        left = "[0" if include_zero else "(0"
        raise ValueError(f"production job {key} must be in {left}, 1]")
    return value


def _date_pair(payload: Mapping[str, Any], key: str) -> tuple[str, str]:
    values = payload.get(key)
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError(f"production job {key} must contain two ISO dates")
    parsed: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"production job {key} must contain two ISO dates")
        try:
            date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(
                f"production job {key} must contain two ISO dates"
            ) from error
        parsed.append(value)
    if parsed[0] > parsed[1]:
        raise ValueError(f"production job {key} date range is invalid")
    return parsed[0], parsed[1]


def _json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value


def _bounded_job_chunks(
    open_dates: pd.DatetimeIndex,
    calculation_dates: pd.DatetimeIndex,
    *,
    max_lookback: int,
    available_start: pd.Timestamp,
) -> tuple[YearChunk, ...]:
    chunks: list[YearChunk] = []
    for year in sorted(set(calculation_dates.year)):
        year_dates = calculation_dates[calculation_dates.year == year]
        first = pd.Timestamp(year_dates.min())
        last = pd.Timestamp(year_dates.max())
        first_position = int(open_dates.get_indexer([first])[0])
        required_start = pd.Timestamp(
            open_dates[max(0, first_position - max_lookback)]
        )
        load_start = max(required_start, pd.Timestamp(available_start).normalize())
        chunks.append(
            YearChunk(
                year=int(year),
                calculation_start=first,
                calculation_end=last,
                load_start=load_start,
                load_end=last,
                max_lookback=max_lookback,
                forward_tail=0,
            )
        )
    return tuple(chunks)


def _resolve_job_symbols(
    job: ProductionJob,
) -> tuple[tuple[str, ...], dict[str, str]]:
    master_codes = tuple(
        sorted(
            pd.read_parquet(job.data.stock_master, columns=["ts_code"])["ts_code"]
            .dropna()
            .astype(str)
            .unique()
        )
    )
    if not master_codes:
        raise ValueError("production job stock master contains no symbols")
    symbols = master_codes
    index_views: dict[str, str] = {}
    if job.index_code is None:
        return symbols, index_views
    assert job.data.index_membership is not None
    membership_frame = pd.read_parquet(
        job.data.index_membership,
        columns=["index_code", "effective_date", "con_code"],
        filters=[("index_code", "==", job.index_code)],
    )
    if job.evaluation.symbol_cap is None:
        members = set(membership_frame["con_code"].astype(str))
    else:
        effective = pd.to_datetime(membership_frame["effective_date"], errors="coerce")
        eligible = membership_frame.loc[
            effective.le(pd.Timestamp(job.evaluation.end))
        ].copy()
        if eligible.empty:
            raise ValueError(
                "production job index membership has no effective snapshot"
            )
        latest = pd.to_datetime(eligible["effective_date"]).max()
        members = set(
            eligible.loc[
                pd.to_datetime(eligible["effective_date"]).eq(latest), "con_code"
            ].astype(str)
        )
    symbols = tuple(code for code in master_codes if code in members)
    if not symbols:
        raise ValueError("production job index membership has no master symbols")
    index_views = {job.view: job.index_code}
    return symbols, index_views


def _job_trading_dates(
    job: ProductionJob,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    calendar = pd.read_csv(job.data.trade_calendar)
    required = {"exchange", "cal_date", "is_open"}
    missing = sorted(required.difference(calendar.columns))
    if missing:
        raise ValueError(f"production trade calendar is missing: {missing}")
    dates = pd.to_datetime(
        calendar["cal_date"].astype("string").str.replace(r"\.0$", "", regex=True),
        format="%Y%m%d",
        errors="coerce",
    )
    exchange = calendar["exchange"].isin(["SSE", "SZSE"])
    coverage = pd.DatetimeIndex(dates.loc[exchange]).dropna()
    if (
        coverage.empty
        or coverage.min() > pd.Timestamp(job.evaluation.start)
        or coverage.max() < pd.Timestamp(job.evaluation.end)
    ):
        raise ValueError("production trade calendar does not cover the job range")
    open_dates = (
        pd.DatetimeIndex(
            dates.loc[
                exchange
                & pd.to_numeric(calendar["is_open"], errors="coerce").eq(1)
                & dates.le(pd.Timestamp(job.evaluation.end))
            ]
        )
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    calculation_dates = open_dates[
        (open_dates >= pd.Timestamp(job.evaluation.start))
        & (open_dates <= pd.Timestamp(job.evaluation.end))
    ]
    if calculation_dates.empty:
        raise ValueError("production job evaluation range has no open trading dates")
    return open_dates, calculation_dates


def _peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024
