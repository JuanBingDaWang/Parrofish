"""Chinese-capable application font selection tests."""

from __future__ import annotations

from writing_factory.ui.theme import configure_application_font


def test_configures_nonempty_application_font(qapp) -> None:
    family = configure_application_font(qapp)

    assert family
    assert qapp.font().family() == family
