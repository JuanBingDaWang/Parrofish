"""Short prompt aliases for deterministic information-gap references."""

from __future__ import annotations

from dataclasses import dataclass

from writing_factory.distill.extraction import StructuredDistillationError


@dataclass(frozen=True, slots=True)
class GapReferenceAliases:
    """Expose compact G001-style IDs to the model and restore stable IDs in code."""

    alias_to_gap_id: dict[str, str]
    gap_id_to_alias: dict[str, str]

    @classmethod
    def from_registry(
        cls,
        gap_registry: dict[str, dict[str, object]],
    ) -> GapReferenceAliases:
        pairs = [
            (f"G{index:03d}", gap_id)
            for index, gap_id in enumerate(sorted(gap_registry), 1)
        ]
        return cls(
            alias_to_gap_id=dict(pairs),
            gap_id_to_alias={gap_id: alias for alias, gap_id in pairs},
        )

    def encode_bundle(self, bundle: dict[str, object]) -> dict[str, object]:
        """Replace only local gap IDs in a shallow-copied reducer bundle."""

        encoded = dict(bundle)
        local_gaps: list[dict[str, object]] = []
        raw_gaps = bundle.get("local_information_gaps", [])
        if isinstance(raw_gaps, list):
            for raw_gap in raw_gaps:
                if not isinstance(raw_gap, dict):
                    continue
                gap = dict(raw_gap)
                gap_id = str(gap.get("gap_id", ""))
                alias = self.gap_id_to_alias.get(gap_id)
                if alias is None:
                    raise StructuredDistillationError(
                        "程序生成的信息缺口登记表与短别名映射不一致"
                    )
                gap["gap_id"] = alias
                local_gaps.append(gap)
        encoded["local_information_gaps"] = local_gaps
        return encoded

    def decode(self, identifiers: list[str]) -> list[str]:
        """Restore aliases while accepting already-valid stable IDs for old caches."""

        decoded: list[str] = []
        unknown: list[str] = []
        for identifier in identifiers:
            if identifier in self.alias_to_gap_id:
                decoded.append(self.alias_to_gap_id[identifier])
            elif identifier in self.gap_id_to_alias:
                decoded.append(identifier)
            else:
                unknown.append(identifier)
        if unknown:
            values = ", ".join(sorted(set(unknown)))
            raise StructuredDistillationError(
                f"归并结果引用了未知缺口短标识：{values}。"
                "supporting_gap_ids 只能逐字复制输入中形如 G001 的短标识；"
                f"本次共登记 {len(self.alias_to_gap_id)} 个合法短标识"
            )
        return decoded
