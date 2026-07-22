"""Compile parsed factor expressions into typed, canonical identities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from ashare_factor_backtest.contracts import PriceBasis
from ashare_factor_backtest.expression.catalog import FieldCatalog, OperatorCatalog
from ashare_factor_backtest.expression.errors import ExpressionError
from ashare_factor_backtest.expression.model import (
    CallNode,
    CompiledExpression,
    ConstantNode,
    ExprNode,
    FieldNode,
    ValueType,
)
from ashare_factor_backtest.expression.parser import parse_expression


HFQ_CROSS_SECTION_WARNING = "hfq_price_level_is_not_cross_sectionally_comparable"


@dataclass(frozen=True)
class _Semantic:
    node: ExprNode
    value_type: ValueType
    lookback: int
    complexity: int
    price_basis: PriceBasis | None
    unit_lineage: str
    fields: frozenset[str]
    warnings: tuple[str, ...] = ()
    hfq_scale_sensitive: bool = False


def compile_expression(
    text: str, operators: OperatorCatalog, fields: FieldCatalog
) -> CompiledExpression:
    parsed = parse_expression(text)
    semantic = _compile_node(parsed, operators, fields, text)
    warnings = list(semantic.warnings)
    if semantic.hfq_scale_sensitive:
        warnings.append(HFQ_CROSS_SECTION_WARNING)
    warnings = list(dict.fromkeys(warnings))
    canonical = canonical_expression(semantic.node)
    payload = json.dumps(
        {
            "canonical": canonical,
            "operator_catalog_version": operators.version,
            "field_catalog_version": fields.version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    factor_id = "f_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return CompiledExpression(
        semantic.node,
        canonical,
        factor_id,
        semantic.value_type,
        semantic.lookback,
        semantic.complexity,
        semantic.price_basis,
        semantic.unit_lineage,
        tuple(sorted(semantic.fields)),
        tuple(warnings),
    )


def canonical_expression(node: ExprNode) -> str:
    if isinstance(node, FieldNode):
        return node.name
    if isinstance(node, ConstantNode):
        return str(node.value)
    arguments = [canonical_expression(argument) for argument in node.arguments]
    return f"{node.operator}({','.join(arguments)})"


def explain_expression(
    text: str, operators: OperatorCatalog, fields: FieldCatalog
) -> dict[str, object]:
    compiled = compile_expression(text, operators, fields)
    return {
        "factor_id": compiled.factor_id,
        "canonical": compiled.canonical,
        "output_type": compiled.output_type.value,
        "lookback": compiled.lookback,
        "complexity": compiled.complexity,
        "price_basis": compiled.price_basis.value if compiled.price_basis else None,
        "unit_lineage": compiled.unit_lineage,
        "fields": list(compiled.fields),
        "warnings": list(compiled.warnings),
    }


def _compile_node(
    node: ExprNode, operators: OperatorCatalog, fields: FieldCatalog, text: str
) -> _Semantic:
    if isinstance(node, FieldNode):
        spec = fields.resolve(node.name)
        return _Semantic(
            node, spec.value_type, 0, 0, spec.price_basis, spec.unit_lineage,
            frozenset((spec.name,)),
            hfq_scale_sensitive=(
                spec.price_basis is PriceBasis.HFQ_PIT
                and spec.unit_lineage == "price"
            ),
        )
    if isinstance(node, ConstantNode):
        value_type = (
            ValueType.SCALAR_INT
            if isinstance(node.value, int)
            else ValueType.SCALAR_FLOAT
        )
        return _Semantic(node, value_type, 0, 0, None, "scalar", frozenset())

    spec = operators.resolve(node.operator)
    if len(node.arguments) != len(spec.input_types):
        raise ExpressionError(
            "TYPE_MISMATCH",
            f"{spec.name} expects {len(spec.input_types)} arguments, got {len(node.arguments)}",
            text,
        )
    children = tuple(_compile_node(arg, operators, fields, text) for arg in node.arguments)
    for index, (child, expected) in enumerate(zip(children, spec.input_types, strict=True)):
        if not spec.accepts(index, child.value_type):
            allowed = expected if isinstance(expected, tuple) else (expected,)
            raise ExpressionError(
                "TYPE_MISMATCH",
                f"{spec.name} argument {index} expects "
                f"{'|'.join(value.value for value in allowed)}, got {child.value_type.value}",
                text,
            )
    if spec.requires_panel_input and not any(
        child.value_type is ValueType.PANEL_FLOAT for child in children
    ):
        raise ExpressionError(
            "TYPE_MISMATCH", f"{spec.name} requires at least one panel input", text
        )
    for item in spec.parameter_domain:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"invalid parameter_domain metadata for {spec.name}")
        index, allowed = item
        child = children[int(index)]
        if not isinstance(child.node, ConstantNode) or child.node.value not in allowed:
            raise ExpressionError(
                "INVALID_PARAMETER", f"{spec.name} argument {index} is outside allowed domain", text
            )

    bases = {child.price_basis for child in children if child.price_basis is not None}
    if len(bases) > 1:
        raise ExpressionError("PRICE_BASIS_MISMATCH", "expression mixes price bases", text)
    unit = _output_unit(spec.name, children, text)
    basis = next(iter(bases), None)
    canonical_children = list(children)
    if spec.commutative:
        panel_positions = [
            index
            for index in range(len(spec.input_types))
            if spec.accepts(index, ValueType.PANEL_FLOAT)
        ]
        sorted_panels = sorted(
            (canonical_children[index] for index in panel_positions),
            key=lambda child: canonical_expression(child.node),
        )
        for index, child in zip(panel_positions, sorted_panels, strict=True):
            canonical_children[index] = child
    normalized = CallNode(spec.name, tuple(child.node for child in canonical_children))
    lookback = _lookback(spec.lookback_rule, children)
    warnings = [warning for child in children for warning in child.warnings]
    if spec.category == "cross_section" and any(
        child.hfq_scale_sensitive for child in children
    ):
        warnings.append(HFQ_CROSS_SECTION_WARNING)
    return _Semantic(
        normalized,
        spec.output_type,
        lookback,
        spec.complexity_cost + sum(child.complexity for child in children),
        basis,
        unit,
        frozenset().union(*(child.fields for child in children)),
        tuple(dict.fromkeys(warnings)),
        _hfq_scale_sensitive(spec.name, spec.category, children),
    )


def _hfq_scale_sensitive(
    name: str,
    category: str,
    children: tuple[_Semantic, ...],
) -> bool:
    sensitive = tuple(child.hfq_scale_sensitive for child in children)
    if not any(sensitive):
        return False
    if category == "cross_section":
        return False
    if name in {
        "sign",
        "ts_pct_change",
        "ts_rank",
        "ts_zscore",
        "ts_corr",
        "ts_scale",
        "ts_skew",
        "ts_kurt",
        "ts_r2",
        "ts_argmin",
        "ts_argmax",
    }:
        return False
    if name == "div":
        panel_children = tuple(
            child for child in children if child.value_type is ValueType.PANEL_FLOAT
        )
        if (
            len(panel_children) == 2
            and all(child.hfq_scale_sensitive for child in panel_children)
            and panel_children[0].unit_lineage == panel_children[1].unit_lineage
        ):
            return False
    if name == "ts_beta":
        panel_children = tuple(
            child for child in children if child.value_type is ValueType.PANEL_FLOAT
        )
        if (
            len(panel_children) == 2
            and all(child.hfq_scale_sensitive for child in panel_children)
            and panel_children[0].unit_lineage == panel_children[1].unit_lineage
        ):
            return False
    return True


def _lookback(rule: str, children: tuple[_Semantic, ...]) -> int:
    base = max((child.lookback for child in children), default=0)
    if rule == "max_children":
        return base
    if rule == "stateful":
        return base
    if rule in {"delay", "rolling"}:
        window_node = children[-1].node
        if not isinstance(window_node, ConstantNode) or not isinstance(window_node.value, int):
            raise ValueError("window lookback requires integer constant")
        return base + window_node.value - (1 if rule == "rolling" else 0)
    raise ValueError(f"unknown lookback rule: {rule}")


def _output_unit(name: str, children: tuple[_Semantic, ...], text: str) -> str:
    panel_units = [child.unit_lineage for child in children if child.value_type is ValueType.PANEL_FLOAT]
    if name in {
        "add", "sub", "panel_min", "panel_max", "gt", "ge", "lt", "le"
    } and len(set(panel_units)) > 1:
        raise ExpressionError("UNIT_MISMATCH", f"{name} requires compatible units", text)
    if name == "where":
        panel_branches = [
            children[index].unit_lineage
            for index in (1, 2)
            if children[index].value_type is ValueType.PANEL_FLOAT
        ]
        if len(set(panel_branches)) > 1:
            raise ExpressionError("UNIT_MISMATCH", "where branches require compatible units", text)
        return panel_branches[0] if panel_branches else "ratio"
    if name == "trade_when":
        return children[1].unit_lineage
    if name == "cs_residual":
        return children[0].unit_lineage
    if name in {"gt", "ge", "lt", "le"}:
        return "boolean"
    if name in {
        "sign", "cs_rank", "cs_zscore", "ts_pct_change", "ts_rank", "ts_zscore",
        "ts_corr", "ts_scale", "ts_skew", "ts_kurt", "ts_r2"
    }:
        return "ratio"
    if name == "signed_power":
        exponent = children[1].node
        if not isinstance(exponent, ConstantNode):
            raise ValueError("signed_power exponent must be constant")
        return f"nonlinear({children[0].unit_lineage},{exponent.value})"
    if name == "ts_slope":
        return f"{panel_units[0]}/observation"
    if name == "ts_product":
        return "ratio" if panel_units[0] == "ratio" else f"product({panel_units[0]})"
    if name in {"ts_argmin", "ts_argmax"}:
        return "observation_offset"
    if name == "ts_beta" and len(panel_units) == 2:
        return "ratio" if panel_units[0] == panel_units[1] else f"{panel_units[0]}/{panel_units[1]}"
    if name == "ts_cov" and len(panel_units) == 2:
        return "*".join(sorted(panel_units))
    if name in {
        "add", "sub", "panel_min", "panel_max", "neg", "abs", "signed_log", "signed_sqrt", "cs_demean",
        "cs_winsorize", "ts_delay", "ts_delta", "ts_sum", "ts_mean", "ts_std",
        "ts_min", "ts_max", "ts_ema", "ts_decay_linear",
    }:
        return panel_units[0] if panel_units else "scalar"
    if name == "div":
        left, right = children
        if right.value_type in {ValueType.SCALAR_INT, ValueType.SCALAR_FLOAT}:
            return left.unit_lineage
        if left.value_type in {ValueType.SCALAR_INT, ValueType.SCALAR_FLOAT}:
            return f"1/{right.unit_lineage}"
        return "ratio" if left.unit_lineage == right.unit_lineage else f"{left.unit_lineage}/{right.unit_lineage}"
    if name == "mul":
        return "*".join(sorted(unit for unit in panel_units)) if len(panel_units) == 2 else panel_units[0]
    return panel_units[0] if panel_units else "ratio"
