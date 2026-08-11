from pathlib import PurePosixPath

import pytest
import yaml

from scripts import make_redirect_pages


def test_external_blog_redirect_uses_canonical_url():
    target = "https://qmcsoftware.org/blog/why-add-q-to-mc/"

    page = make_redirect_pages.render_page(
        PurePosixPath("2020/06/25/why_add_q_to_mc"),
        make_redirect_pages.validate_target(target),
        "Why Add Q to MC?",
    )

    assert f'<meta http-equiv="refresh" content="0; url={target}">' in page
    assert f"window.location.replace(\"{target}\")" in page
    assert f"[Why Add Q to MC?]({target})" in page


@pytest.mark.parametrize(
    "target",
    [
        "http://qmcsoftware.org/blog/example/",
        "https://example.com/blog/example/",
        "https://qmcsoftware.org/software/",
        "https://qmcsoftware.org/blog/example/?preview=1",
    ],
)
def test_external_redirect_rejects_unapproved_targets(target):
    with pytest.raises(ValueError, match="external redirect target"):
        make_redirect_pages.validate_target(target)


def test_internal_redirect_target_remains_supported():
    assert make_redirect_pages.validate_target("/community/") == PurePosixPath(
        "community"
    )


def test_manifest_generates_all_external_blog_redirects():
    manifest = yaml.safe_load(
        make_redirect_pages.MANIFEST.read_text(encoding="utf-8")
    )
    blog_redirects = [
        entry
        for entry in manifest["redirects"]
        if entry["target"].startswith("https://qmcsoftware.org/blog/")
    ]

    assert len(blog_redirects) == 18
    for entry in blog_redirects:
        source = make_redirect_pages.validate_path(entry["source"], "source")
        page = make_redirect_pages.DOCS.joinpath(*source.parts, "index.md").read_text(
            encoding="utf-8"
        )
        assert page.count(entry["target"]) == 3
