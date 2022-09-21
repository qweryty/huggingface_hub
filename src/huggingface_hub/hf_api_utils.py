# TODO: remove after deprecation period is over (v0.10)
import os
import re
import subprocess
import warnings
from typing import List, Optional, Tuple, Union

from ._commit_api import CommitOperationAdd
from .constants import ENDPOINT, REPO_TYPES, REPO_TYPES_MAPPING
from .utils import filter_repo_objects


_REGEX_DISCUSSION_URL = re.compile(r".*/discussions/(\d+)$")


def _validate_repo_id_deprecation(repo_id, name, organization):
    """Returns (name, organization) from the input."""
    if repo_id and not name and organization:
        # this means the user had passed name as positional, now mapped to
        # repo_id and is passing organization as well. This wouldn't be an
        # issue if they pass everything as kwarg. So we switch the parameters
        # here:
        repo_id, name = name, repo_id

    if not (repo_id or name):
        raise ValueError(
            "No name provided. Please pass `repo_id` with a valid repository name."
        )

    if repo_id and (name or organization):
        raise ValueError(
            "Only pass `repo_id` and leave deprecated `name` and `organization` to be"
            " None."
        )
    elif name or organization:
        warnings.warn(
            "`name` and `organization` input arguments are deprecated and "
            "will be removed in v0.10. Pass `repo_id` instead.",
            FutureWarning,
        )
    else:
        if "/" in repo_id:
            organization, name = repo_id.split("/")
        else:
            organization, name = None, repo_id
    return name, organization


def repo_type_and_id_from_hf_id(
    hf_id: str, hub_url: Optional[str] = None
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Returns the repo type and ID from a huggingface.co URL linking to a
    repository

    Args:
        hf_id (`str`):
            An URL or ID of a repository on the HF hub. Accepted values are:

            - https://huggingface.co/<repo_type>/<namespace>/<repo_id>
            - https://huggingface.co/<namespace>/<repo_id>
            - <repo_type>/<namespace>/<repo_id>
            - <namespace>/<repo_id>
            - <repo_id>
        hub_url (`str`, *optional*):
            The URL of the HuggingFace Hub, defaults to https://huggingface.co
    """
    hub_url = re.sub(r"https?://", "", hub_url if hub_url is not None else ENDPOINT)
    is_hf_url = hub_url in hf_id and "@" not in hf_id
    url_segments = hf_id.split("/")
    is_hf_id = len(url_segments) <= 3

    if is_hf_url:
        namespace, repo_id = url_segments[-2:]
        if namespace == hub_url:
            namespace = None
        if len(url_segments) > 2 and hub_url not in url_segments[-3]:
            repo_type = url_segments[-3]
        else:
            repo_type = None
    elif is_hf_id:
        if len(url_segments) == 3:
            # Passed <repo_type>/<user>/<model_id> or <repo_type>/<org>/<model_id>
            repo_type, namespace, repo_id = url_segments[-3:]
        elif len(url_segments) == 2:
            # Passed <user>/<model_id> or <org>/<model_id>
            namespace, repo_id = hf_id.split("/")[-2:]
            repo_type = None
        else:
            # Passed <model_id>
            repo_id = url_segments[0]
            namespace, repo_type = None, None
    else:
        raise ValueError(
            f"Unable to retrieve user and repo ID from the passed HF ID: {hf_id}"
        )

    repo_type = (
        repo_type if repo_type in REPO_TYPES else REPO_TYPES_MAPPING.get(repo_type)
    )

    return repo_type, namespace, repo_id


def write_to_credential_store(username: str, password: str):
    with subprocess.Popen(
        "git credential-store store".split(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as process:
        input_username = f"username={username.lower()}"
        input_password = f"password={password}"

        process.stdin.write(
            f"url={ENDPOINT}\n{input_username}\n{input_password}\n\n".encode("utf-8")
        )
        process.stdin.flush()


def read_from_credential_store(
    username=None,
) -> Tuple[Union[str, None], Union[str, None]]:
    """
    Reads the credential store relative to huggingface.co. If no `username` is
    specified, will read the first entry for huggingface.co, otherwise will read
    the entry corresponding to the username specified.

    The username returned will be all lowercase.
    """
    with subprocess.Popen(
        "git credential-store get".split(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as process:
        standard_input = f"url={ENDPOINT}\n"

        if username is not None:
            standard_input += f"username={username.lower()}\n"

        standard_input += "\n"

        process.stdin.write(standard_input.encode("utf-8"))
        process.stdin.flush()
        output = process.stdout.read()
        output = output.decode("utf-8")

    if len(output) == 0:
        return None, None

    username, password = [line for line in output.split("\n") if len(line) != 0]
    return username.split("=")[1], password.split("=")[1]


def erase_from_credential_store(username=None):
    """
    Erases the credential store relative to huggingface.co. If no `username` is
    specified, will erase the first entry for huggingface.co, otherwise will
    erase the entry corresponding to the username specified.
    """
    with subprocess.Popen(
        "git credential-store erase".split(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as process:
        standard_input = f"url={ENDPOINT}\n"

        if username is not None:
            standard_input += f"username={username.lower()}\n"

        standard_input += "\n"

        process.stdin.write(standard_input.encode("utf-8"))
        process.stdin.flush()


def _parse_revision_from_pr_url(pr_url: str) -> str:
    """Safely parse revision number from a PR url.

    Example:
    ```py
    >>> _parse_revision_from_pr_url("https://huggingface.co/bigscience/bloom/discussions/2")
    "refs/pr/2"
    ```
    """
    re_match = re.match(_REGEX_DISCUSSION_URL, pr_url)
    if re_match is None:
        raise RuntimeError(
            "Unexpected response from the hub, expected a Pull Request URL but got:"
            f" '{pr_url}'"
        )
    return f"refs/pr/{re_match[1]}"


def _prepare_upload_folder_commit(
    folder_path: str,
    path_in_repo: str,
    allow_patterns: Optional[Union[List[str], str]] = None,
    ignore_patterns: Optional[Union[List[str], str]] = None,
) -> List[CommitOperationAdd]:
    """Generate the list of Add operations for a commit to upload a folder.

    Files not matching the `allow_patterns` (allowlist) and `ignore_patterns` (denylist)
    constraints are discarded.
    """
    folder_path = os.path.normpath(os.path.expanduser(folder_path))
    if not os.path.isdir(folder_path):
        raise ValueError(f"Provided path: '{folder_path}' is not a directory")

    files_to_add: List[CommitOperationAdd] = []
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, folder_path)
            files_to_add.append(
                CommitOperationAdd(
                    path_or_fileobj=abs_path,
                    path_in_repo=os.path.normpath(
                        os.path.join(path_in_repo, rel_path)
                    ).replace(os.sep, "/"),
                )
            )

    return list(
        filter_repo_objects(
            files_to_add,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            key=lambda x: x.path_in_repo,
        )
    )
