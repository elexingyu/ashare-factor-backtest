"""Versioned metadata and callable registry for the frozen v2 operator set."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Mapping

from ashare_factor_backtest.expression.catalog import OperatorCatalog
from ashare_factor_backtest.expression.model import OperatorSpec, TypeConstraint, ValueType
from ashare_factor_backtest.expression.operators import arithmetic, conditional, cross_section, pairwise, time_series


WINDOWS = (1, 2, 3, 5, 10, 20, 40, 60, 120, 252)


def build_operator_catalog() -> tuple[OperatorCatalog, Mapping[str, Callable[..., object]]]:
    return _build_operator_catalog(
        "astock_formula_ops_v3", include_stateful=False, include_path=False
    )


def build_stateful_operator_catalog() -> tuple[OperatorCatalog, Mapping[str, Callable[..., object]]]:
    return _build_operator_catalog(
        "astock_formula_ops_v4_stateful", include_stateful=True, include_path=False
    )


def build_path_operator_catalog() -> tuple[OperatorCatalog, Mapping[str, Callable[..., object]]]:
    return _build_operator_catalog(
        "astock_formula_ops_v5_path", include_stateful=False, include_path=True
    )


def build_production_operator_catalog() -> tuple[
    OperatorCatalog, Mapping[str, Callable[..., object]]
]:
    return _build_operator_catalog(
        "astock_formula_ops_v6_production",
        include_stateful=False,
        include_path=False,
        rank_function=cross_section.cs_rank_stable,
    )


def build_production_path_operator_catalog() -> tuple[
    OperatorCatalog, Mapping[str, Callable[..., object]]
]:
    return _build_operator_catalog(
        "astock_formula_ops_v7_production_path",
        include_stateful=False,
        include_path=True,
        rank_function=cross_section.cs_rank_stable,
    )


def resolve_production_operator_catalog(
    version: str,
) -> Callable[[], tuple[OperatorCatalog, Mapping[str, Callable[..., object]]]]:
    builders = {
        "astock_formula_ops_v6_production": build_production_operator_catalog,
        "astock_formula_ops_v7_production_path": build_production_path_operator_catalog,
    }
    try:
        return builders[version]
    except KeyError as error:
        raise ValueError(f"unsupported production operator catalog: {version}") from error


def _build_operator_catalog(
    version: str,
    *,
    include_stateful: bool,
    include_path: bool,
    rank_function: Callable[..., object] = cross_section.cs_rank,
) -> tuple[OperatorCatalog, Mapping[str, Callable[..., object]]]:
    catalog = OperatorCatalog(version)
    functions: dict[str, Callable[..., object]] = {}

    def add(spec: OperatorSpec, function: Callable[..., object]) -> None:
        catalog.register(spec)
        functions[spec.name] = function

    panel = ValueType.PANEL_FLOAT
    integer = ValueType.SCALAR_INT
    scalar = ValueType.SCALAR_FLOAT
    boolean = ValueType.PANEL_BOOL
    number = (panel, integer, scalar)
    for name, aliases, function, commutative in (
        ("add", (), arithmetic.add, True), ("sub", (), arithmetic.sub, False),
        ("mul", (), arithmetic.mul, True), ("div", (), arithmetic.div, False),
    ):
        add(
            _spec(
                name, aliases, "arithmetic", (number, number), panel,
                commutative=commutative, requires_panel=True,
            ),
            function,
        )
    for name, aliases, function in (
        ("neg", (), arithmetic.neg), ("abs", ("absolute",), arithmetic.abs_value),
        ("signed_log", (), arithmetic.signed_log), ("signed_sqrt", (), arithmetic.signed_sqrt),
        ("sign", (), arithmetic.sign),
    ):
        add(_spec(name, aliases, "arithmetic", (panel,), panel), function)
    add(
        _spec(
            "signed_power", (), "arithmetic", (panel, (integer, scalar)), panel,
            domain=((1, (0.5, 1.5, 2, 3)),),
        ),
        arithmetic.signed_power,
    )
    for name, function in (("panel_min", arithmetic.panel_min), ("panel_max", arithmetic.panel_max)):
        add(
            _spec(
                name, (), "arithmetic", (number, number), panel,
                commutative=True, requires_panel=True,
            ),
            function,
        )

    for name, aliases, function in (
        ("cs_rank", ("rank",), rank_function),
        ("cs_zscore", ("zscore",), cross_section.cs_zscore),
        ("cs_demean", ("demean",), cross_section.cs_demean),
    ):
        add(_spec(name, aliases, "cross_section", (panel,), panel), function)
    add(
        _spec(
            "cs_winsorize", ("winsorize",), "cross_section", (panel, scalar), panel,
            domain=((1, (0.01, 0.05)),),
        ),
        cross_section.cs_winsorize,
    )
    add(
        _spec(
            "cs_residual", (), "cross_section", (panel, panel), panel,
        ),
        cross_section.cs_residual,
    )

    unary_windows = (
        ("ts_delay", ("delay", "ref"), time_series.ts_delay, "delay"),
        ("ts_delta", ("delta",), time_series.ts_delta, "delay"),
        ("ts_pct_change", ("pct_change",), time_series.ts_pct_change, "delay"),
        ("ts_sum", (), time_series.ts_sum, "rolling"),
        ("ts_mean", ("mean",), time_series.ts_mean, "rolling"),
        ("ts_std", ("std",), time_series.ts_std, "rolling"),
        ("ts_min", (), time_series.ts_min, "rolling"),
        ("ts_max", (), time_series.ts_max, "rolling"),
        ("ts_rank", (), time_series.ts_rank, "rolling"),
        ("ts_argmin", (), time_series.ts_argmin, "rolling"),
        ("ts_argmax", (), time_series.ts_argmax, "rolling"),
        ("ts_zscore", (), time_series.ts_zscore, "rolling"),
        ("ts_ema", ("ema",), time_series.ts_ema, "rolling"),
        ("ts_decay_linear", ("decay_linear",), time_series.ts_decay_linear, "rolling"),
        ("ts_scale", (), time_series.ts_scale, "rolling"),
        ("ts_skew", (), time_series.ts_skew, "rolling"),
        ("ts_kurt", (), time_series.ts_kurt, "rolling"),
        ("ts_slope", (), time_series.ts_slope, "rolling"),
        ("ts_r2", (), time_series.ts_r2, "rolling"),
        ("ts_product", (), time_series.ts_product, "rolling"),
    )
    for name, aliases, function, rule in unary_windows:
        add(
            _spec(
                name, aliases, "time_series", (panel, integer), panel,
                domain=((1, WINDOWS),), rule=rule,
            ),
            function,
        )
    if include_path:
        for name, function in (
            ("ts_path_efficiency", time_series.ts_path_efficiency),
            ("ts_turning_rate", time_series.ts_turning_rate),
            ("ts_signed_run_length", time_series.ts_signed_run_length),
        ):
            add(
                _spec(
                    name,
                    (),
                    "path",
                    (panel, integer),
                    panel,
                    domain=((1, WINDOWS),),
                    rule="rolling",
                ),
                function,
            )

    for name, aliases, function in (
        ("ts_corr", ("correlation",), pairwise.ts_corr),
        ("ts_cov", ("covariance",), pairwise.ts_cov),
        ("ts_beta", ("beta",), pairwise.ts_beta),
    ):
        add(
            _spec(
                name, aliases, "pairwise", (panel, panel, integer), panel,
                domain=((2, WINDOWS),), rule="rolling",
                commutative=name in {"ts_corr", "ts_cov"},
            ),
            function,
        )

    for name, function in (("gt", conditional.gt), ("ge", conditional.ge), ("lt", conditional.lt), ("le", conditional.le)):
        add(
            _spec(
                name, (), "comparison", (number, number), boolean,
                requires_panel=True,
            ),
            function,
        )
    add(_spec("where", (), "conditional", (boolean, number, number), panel), conditional.where)
    if include_stateful:
        add(
            OperatorSpec(
                name="trade_when",
                aliases=(),
                version="1",
                category="stateful",
                input_types=(boolean, panel, boolean),
                output_type=panel,
                causal_contract="forward_state_only_entry_priority",
                nan_policy="missing_conditions_false_entry_alpha_preserved",
                parameter_domain=(),
                lookback_rule="stateful",
                commutative=False,
                complexity_cost=2,
                examples=("trade_when(gt(close,open),close,lt(close,open))",),
            ),
            conditional.trade_when,
        )
    return catalog, functions


def write_operator_catalog(path: Path) -> None:
    catalog, _ = build_operator_catalog()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(catalog.export_payload(), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _spec(
    name: str,
    aliases: tuple[str, ...],
    category: str,
    inputs: tuple[TypeConstraint, ...],
    output: ValueType,
    *,
    domain: tuple[object, ...] = (),
    rule: str = "max_children",
    commutative: bool = False,
    requires_panel: bool = False,
) -> OperatorSpec:
    return OperatorSpec(
        name=name,
        aliases=aliases,
        version="1",
        category=category,
        input_types=inputs,
        output_type=output,
        causal_contract="same_date_or_backward_only",
        nan_policy="exclude_from_statistics_and_preserve",
        parameter_domain=domain,
        lookback_rule=rule,
        commutative=commutative,
        complexity_cost=1,
        examples=(_example(name, inputs, domain),),
        requires_panel_input=requires_panel,
    )


def _example(
    name: str, inputs: tuple[TypeConstraint, ...], domain: tuple[object, ...]
) -> str:
    arguments = []
    panels_seen = 0
    domains = {int(index): tuple(allowed) for index, allowed in domain}
    for index, constraint in enumerate(inputs):
        allowed = constraint if isinstance(constraint, tuple) else (constraint,)
        value_type = allowed[0]
        if index in domains:
            arguments.append(str(domains[index][0]))
            continue
        if value_type is ValueType.PANEL_FLOAT:
            arguments.append("close" if panels_seen == 0 else "open")
            panels_seen += 1
        elif value_type is ValueType.PANEL_BOOL:
            arguments.append("gt(close,open)")
        elif value_type is ValueType.SCALAR_INT:
            arguments.append("20")
        else:
            arguments.append("0.05")
    return f"{name}({','.join(arguments)})"
