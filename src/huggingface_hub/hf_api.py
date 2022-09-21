# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import BinaryIO, Dict, Iterable, Iterator, List, Optional, Tuple, Union

from asgiref.sync import async_to_sync

from ._commit_api import CommitOperation
from .async_hf_api import *  # noqa FIXME
from .async_hf_api import AsyncHfApi
from .community import (
    Discussion,
    DiscussionComment,
    DiscussionStatusChange,
    DiscussionTitleChange,
    DiscussionWithDetails,
)
from .constants import USERNAME_PLACEHOLDER  # noqa
from .hf_api_models import *  # noqa FIXME
from .hf_api_models import DatasetInfo, MetricInfo, ModelInfo, SpaceInfo
from .hf_api_utils import *  # noqa FIXME
from .hf_api_utils import erase_from_credential_store, write_to_credential_store
from .utils import logging, validate_hf_hub_args
from .utils._deprecation import _deprecate_positional_args
from .utils._typing import Literal
from .utils.endpoint_helpers import (
    AttributeDictionary,
    DatasetFilter,
    DatasetTags,
    ModelFilter,
    ModelTags,
)
from .utils.sync import iter_over_async


logger = logging.get_logger(__name__)


class ModelSearchArguments(AttributeDictionary):
    """
    A nested namespace object holding all possible values for properties of
    models currently hosted in the Hub with tab-completion. If a value starts
    with a number, it will only exist in the dictionary

    Example:

    ```python
    >>> args = ModelSearchArguments()
    >>> args.author_or_organization.huggingface
    >>> args.language.en
    ```
    """

    def __init__(self):
        self._api = HfApi()
        tags = self._api.get_model_tags()
        super().__init__(tags)
        self._process_models()

    def _process_models(self):
        def clean(s: str):
            return s.replace(" ", "").replace("-", "_").replace(".", "_")

        models = self._api.list_models()
        author_dict, model_name_dict = AttributeDictionary(), AttributeDictionary()
        for model in models:
            if "/" in model.modelId:
                author, name = model.modelId.split("/")
                author_dict[author] = clean(author)
            else:
                name = model.modelId
            model_name_dict[name] = clean(name)
        self["model_name"] = model_name_dict
        self["author"] = author_dict


class DatasetSearchArguments(AttributeDictionary):
    """
    A nested namespace object holding all possible values for properties of
    datasets currently hosted in the Hub with tab-completion. If a value starts
    with a number, it will only exist in the dictionary

    Example:

    ```python
    >>> args = DatasetSearchArguments()
    >>> args.author_or_organization.huggingface
    >>> args.language.en
    ```
    """

    def __init__(self):
        self._api = HfApi()
        tags = self._api.get_dataset_tags()
        super().__init__(tags)
        self._process_models()

    def _process_models(self):
        def clean(s: str):
            return s.replace(" ", "").replace("-", "_").replace(".", "_")

        datasets = self._api.list_datasets()
        author_dict, dataset_name_dict = AttributeDictionary(), AttributeDictionary()
        for dataset in datasets:
            if "/" in dataset.id:
                author, name = dataset.id.split("/")
                author_dict[author] = clean(author)
            else:
                name = dataset.id
            dataset_name_dict[name] = clean(name)
        self["dataset_name"] = dataset_name_dict
        self["author"] = author_dict


class HfApi:
    def __init__(self, endpoint: Optional[str] = None):
        self._async_api = AsyncHfApi(endpoint=endpoint)

    @property
    def endpoint(self) -> str:
        return self._async_api.endpoint

    def whoami(self, token: Optional[str] = None) -> Dict:
        # TODO something better(class decorator/function to generate?)
        return async_to_sync(self._async_api.whoami)(token=token)

    def get_model_tags(self) -> ModelTags:
        return async_to_sync(self._async_api.get_model_tags)()

    def get_dataset_tags(self) -> DatasetTags:
        return async_to_sync(self._async_api.get_dataset_tags)()

    def list_models(
        self,
        *,
        filter: Union[ModelFilter, str, Iterable[str], None] = None,
        author: Optional[str] = None,
        search: Optional[str] = None,
        emissions_thresholds: Optional[Tuple[float, float]] = None,
        sort: Union[Literal["lastModified"], str, None] = None,
        direction: Optional[Literal[-1]] = None,
        limit: Optional[int] = None,
        full: Optional[bool] = None,
        cardData: Optional[bool] = None,
        fetch_config: Optional[bool] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
    ) -> List[ModelInfo]:
        return async_to_sync(self._async_api.list_models)(
            filter=filter,
            author=author,
            search=search,
            emissions_thresholds=emissions_thresholds,
            sort=sort,
            direction=direction,
            limit=limit,
            full=full,
            card_data=cardData,
            fetch_config=fetch_config,
            use_auth_token=use_auth_token,
        )

    def list_datasets(
        self,
        *,
        filter: Union[DatasetFilter, str, Iterable[str], None] = None,
        author: Optional[str] = None,
        search: Optional[str] = None,
        sort: Union[Literal["lastModified"], str, None] = None,
        direction: Optional[Literal[-1]] = None,
        limit: Optional[int] = None,
        cardData: Optional[bool] = None,
        full: Optional[bool] = None,
        use_auth_token: Optional[str] = None,
    ) -> List[DatasetInfo]:
        return async_to_sync(self._async_api.list_datasets)(
            filter=filter,
            author=author,
            search=search,
            sort=sort,
            direction=direction,
            limit=limit,
            card_data=cardData,
            full=full,
            use_auth_token=use_auth_token,
        )

    def list_metrics(self) -> List[MetricInfo]:
        return async_to_sync(self._async_api.list_metrics)()

    def list_spaces(
        self,
        *,
        filter: Union[str, Iterable[str], None] = None,
        author: Optional[str] = None,
        search: Optional[str] = None,
        sort: Union[Literal["lastModified"], str, None] = None,
        direction: Optional[Literal[-1]] = None,
        limit: Optional[int] = None,
        datasets: Union[str, Iterable[str], None] = None,
        models: Union[str, Iterable[str], None] = None,
        linked: Optional[bool] = None,
        full: Optional[bool] = None,
        use_auth_token: Optional[str] = None,
    ) -> List[SpaceInfo]:
        return async_to_sync(self._async_api.list_spaces)(
            filter=filter,
            author=author,
            search=search,
            sort=sort,
            direction=direction,
            limit=limit,
            datasets=datasets,
            models=models,
            linked=linked,
            full=full,
            use_auth_token=use_auth_token,
        )

    @validate_hf_hub_args
    def model_info(
        self,
        repo_id: str,
        *,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
        securityStatus: Optional[bool] = None,
        files_metadata: bool = False,
        use_auth_token: Optional[Union[bool, str]] = None,
    ) -> ModelInfo:
        return async_to_sync(self._async_api.model_info)(
            repo_id=repo_id,
            revision=revision,
            token=token,
            timeout=timeout,
            security_status=securityStatus,
            files_metadata=files_metadata,
            use_auth_token=use_auth_token,
        )

    @validate_hf_hub_args
    def dataset_info(
        self,
        repo_id: str,
        *,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
        files_metadata: bool = False,
        use_auth_token: Optional[Union[bool, str]] = None,
    ) -> DatasetInfo:
        return async_to_sync(self._async_api.dataset_info)(
            repo_id=repo_id,
            revision=revision,
            token=token,
            timeout=timeout,
            files_metadata=files_metadata,
            use_auth_token=use_auth_token,
        )

    @validate_hf_hub_args
    def space_info(
        self,
        repo_id: str,
        *,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
        files_metadata: bool = False,
        use_auth_token: Optional[Union[bool, str]] = None,
    ) -> SpaceInfo:
        return async_to_sync(self._async_api.space_info)(
            repo_id=repo_id,
            revision=revision,
            token=token,
            timeout=timeout,
            files_metadata=files_metadata,
            use_auth_token=use_auth_token,
        )

    @validate_hf_hub_args
    def repo_info(
        self,
        repo_id: str,
        *,
        revision: Optional[str] = None,
        repo_type: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
        files_metadata: bool = False,
        use_auth_token: Optional[Union[bool, str]] = None,
    ) -> Union[ModelInfo, DatasetInfo, SpaceInfo]:
        return async_to_sync(self._async_api.repo_info)(
            repo_id=repo_id,
            revision=revision,
            repo_type=repo_type,
            token=token,
            timeout=timeout,
            files_metadata=files_metadata,
            use_auth_token=use_auth_token,
        )

    @validate_hf_hub_args
    def list_repo_files(
        self,
        repo_id: str,
        *,
        revision: Optional[str] = None,
        repo_type: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
    ) -> List[str]:
        return async_to_sync(self._async_api.list_repo_files)(repo_id=repo_id)

    @validate_hf_hub_args
    @_deprecate_positional_args(version="0.12")
    def create_repo(
        self,
        repo_id: str = None,
        *,
        token: Optional[str] = None,
        organization: Optional[str] = None,
        private: bool = False,
        repo_type: Optional[str] = None,
        exist_ok: Optional[bool] = False,
        space_sdk: Optional[str] = None,
        name: Optional[str] = None,
    ) -> str:
        return async_to_sync(self._async_api.create_repo)(
            repo_id=repo_id,
            token=token,
            organization=organization,
            private=private,
            repo_type=repo_type,
            exist_ok=exist_ok,
            space_sdk=space_sdk,
            name=name,
        )

    @validate_hf_hub_args
    def delete_repo(
        self,
        repo_id: str = None,
        *,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ):
        async_to_sync(self._async_api.delete_repo)(
            repo_id=repo_id, token=token, repo_type=repo_type
        )

    @validate_hf_hub_args
    def update_repo_visibility(
        self,
        repo_id: str = None,
        private: bool = False,
        *,
        token: Optional[str] = None,
        organization: Optional[str] = None,
        repo_type: Optional[str] = None,
        name: str = None,
    ) -> Dict[str, bool]:
        return async_to_sync(self._async_api.update_repo_visibility)(
            repo_id=repo_id,
            private=private,
            token=token,
            organization=organization,
            repo_type=repo_type,
            name=name,
        )

    def move_repo(
        self,
        from_id: str,
        to_id: str,
        *,
        repo_type: Optional[str] = None,
        token: Optional[str] = None,
    ):
        async_to_sync(self._async_api.move_repo)(
            from_id=from_id, to_id=to_id, repo_type=repo_type, token=token
        )

    @validate_hf_hub_args
    def create_commit(
        self,
        repo_id: str,
        operations: Iterable[CommitOperation],
        *,
        commit_message: str,
        commit_description: Optional[str] = None,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        create_pr: Optional[bool] = None,
        num_threads: int = 5,
        parent_commit: Optional[str] = None,
    ) -> Optional[str]:
        return async_to_sync(self._async_api.create_commit)(
            repo_id=repo_id,
            operations=operations,
            commit_message=commit_message,
            commit_description=commit_description,
            token=token,
            repo_type=repo_type,
            revision=revision,
            create_pr=create_pr,
            num_threads=num_threads,
            parent_commit=parent_commit,
        )

    @validate_hf_hub_args
    def upload_file(
        self,
        *,
        path_or_fileobj: Union[str, bytes, BinaryIO],
        path_in_repo: str,
        repo_id: str,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        identical_ok: Optional[bool] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        create_pr: Optional[bool] = None,
        parent_commit: Optional[str] = None,
    ) -> str:
        return async_to_sync(self._async_api.upload_file)(
            path_or_fileobj=path_or_fileobj,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            token=token,
            repo_type=repo_type,
            revision=revision,
            identical_ok=identical_ok,
            commit_message=commit_message,
            commit_description=commit_description,
            create_pr=create_pr,
            parent_commit=parent_commit,
        )

    @validate_hf_hub_args
    def upload_folder(
        self,
        *,
        repo_id: str,
        folder_path: str,
        path_in_repo: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        create_pr: Optional[bool] = None,
        parent_commit: Optional[str] = None,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
    ) -> str:
        return async_to_sync(self._async_api.upload_folder)(
            repo_id=repo_id,
            folder_path=folder_path,
            path_in_repo=path_in_repo,
            commit_message=commit_message,
            commit_description=commit_description,
            token=token,
            repo_type=repo_type,
            revision=revision,
            create_pr=create_pr,
            parent_commit=parent_commit,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    @validate_hf_hub_args
    def delete_file(
        self,
        path_in_repo: str,
        repo_id: str,
        *,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        create_pr: Optional[bool] = None,
        parent_commit: Optional[str] = None,
    ):
        async_to_sync(self._async_api.delete_file)(
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            token=token,
            repo_type=repo_type,
            revision=revision,
            commit_message=commit_message,
            commit_description=commit_description,
            create_pr=create_pr,
            parent_commit=parent_commit,
        )

    def get_full_repo_name(
        self,
        model_id: str,
        *,
        organization: Optional[str] = None,
        token: Optional[str] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
    ) -> str:
        return async_to_sync(self._async_api.get_full_repo_name)(
            model_id=model_id,
            organization=organization,
            token=token,
            use_auth_token=use_auth_token,
        )

    @validate_hf_hub_args
    def get_repo_discussions(
        self,
        repo_id: str,
        *,
        repo_type: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Iterator[Discussion]:
        return iter_over_async(
            self._async_api.get_repo_discussions(
                repo_id=repo_id, repo_type=repo_type, token=token
            )
        )

    @validate_hf_hub_args
    def get_discussion_details(
        self,
        repo_id: str,
        discussion_num: int,
        *,
        repo_type: Optional[str] = None,
        token: Optional[str] = None,
    ) -> DiscussionWithDetails:
        return async_to_sync(self._async_api.get_discussion_details)(
            repo_id=repo_id,
            discussion_num=discussion_num,
            repo_type=repo_type,
            token=token,
        )

    @validate_hf_hub_args
    def create_discussion(
        self,
        repo_id: str,
        title: str,
        *,
        token: str,
        description: Optional[str] = None,
        repo_type: Optional[str] = None,
        pull_request: bool = False,
    ) -> DiscussionWithDetails:
        return async_to_sync(self._async_api.create_discussion)(
            repo_id=repo_id,
            title=title,
            token=token,
            description=description,
            repo_type=repo_type,
            pull_request=pull_request,
        )

    @validate_hf_hub_args
    def create_pull_request(
        self,
        repo_id: str,
        title: str,
        *,
        token: str,
        description: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> DiscussionWithDetails:
        return async_to_sync(self._async_api.create_pull_request)(
            repo_id=repo_id,
            title=title,
            token=token,
            description=description,
            repo_type=repo_type,
        )

    @validate_hf_hub_args
    def comment_discussion(
        self,
        repo_id: str,
        discussion_num: int,
        comment: str,
        *,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> DiscussionComment:
        return async_to_sync(self._async_api.comment_discussion)(
            repo_id=repo_id,
            discussion_num=discussion_num,
            comment=comment,
            token=token,
            repo_type=repo_type,
        )

    @validate_hf_hub_args
    def rename_discussion(
        self,
        repo_id: str,
        discussion_num: int,
        new_title: str,
        *,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> DiscussionTitleChange:
        return async_to_sync(self._async_api.rename_discussion)(
            repo_id=repo_id,
            discussion_num=discussion_num,
            new_title=new_title,
            token=token,
            repo_type=repo_type,
        )

    @validate_hf_hub_args
    def change_discussion_status(
        self,
        repo_id: str,
        discussion_num: int,
        new_status: Literal["open", "closed"],
        *,
        token: Optional[str] = None,
        comment: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> DiscussionStatusChange:
        return async_to_sync(self._async_api.change_discussion_status)(
            repo_id=repo_id,
            discussion_num=discussion_num,
            new_status=new_status,
            token=token,
            comment=comment,
            repo_type=repo_type,
        )

    @validate_hf_hub_args
    def merge_pull_request(
        self,
        repo_id: str,
        discussion_num: int,
        *,
        token: str,
        comment: Optional[str] = None,
        repo_type: Optional[str] = None,
    ):
        async_to_sync(self._async_api.merge_pull_request)(
            repo_id=repo_id,
            discussion_num=discussion_num,
            token=token,
            comment=comment,
            repo_type=repo_type,
        )

    @validate_hf_hub_args
    def edit_discussion_comment(
        self,
        repo_id: str,
        discussion_num: int,
        comment_id: str,
        new_content: str,
        *,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> DiscussionComment:
        return async_to_sync(self._async_api.edit_discussion_comment)(
            repo_id=repo_id,
            discussion_num=discussion_num,
            comment_id=comment_id,
            new_content=new_content,
            token=token,
            repo_type=repo_type,
        )

    @validate_hf_hub_args
    def hide_discussion_comment(
        self,
        repo_id: str,
        discussion_num: int,
        comment_id: str,
        *,
        token: str,
        repo_type: Optional[str] = None,
    ) -> DiscussionComment:
        return async_to_sync(self._async_api.hide_discussion_comment)(
            repo_id=repo_id,
            discussion_num=discussion_num,
            comment_id=comment_id,
            token=token,
            repo_type=repo_type,
        )

    # FIXME remove
    @staticmethod
    def set_access_token(access_token: str):
        """
        Saves the passed access token so git can correctly authenticate the
        user.

        Args:
            access_token (`str`):
                The access token to save.
        """
        write_to_credential_store(USERNAME_PLACEHOLDER, access_token)

    @staticmethod
    def unset_access_token():
        """
        Resets the user's access token.
        """
        erase_from_credential_store(USERNAME_PLACEHOLDER)

    def _validate_or_retrieve_token(
        self,
        token: Optional[Union[str, bool]] = None,
        name: Optional[str] = None,
        function_name: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        return async_to_sync(self._async_api._validate_or_retrieve_token)(
            token=token, name=name, function_name=function_name
        )


api = HfApi()

set_access_token = api.set_access_token
unset_access_token = api.unset_access_token

whoami = api.whoami

list_models = api.list_models
model_info = api.model_info

list_datasets = api.list_datasets
dataset_info = api.dataset_info

list_spaces = api.list_spaces
space_info = api.space_info

repo_info = api.repo_info
list_repo_files = api.list_repo_files

list_metrics = api.list_metrics

get_model_tags = api.get_model_tags
get_dataset_tags = api.get_dataset_tags

create_commit = api.create_commit
create_repo = api.create_repo
delete_repo = api.delete_repo
update_repo_visibility = api.update_repo_visibility
move_repo = api.move_repo
upload_file = api.upload_file
upload_folder = api.upload_folder
delete_file = api.delete_file
get_full_repo_name = api.get_full_repo_name

get_discussion_details = api.get_discussion_details
get_repo_discussions = api.get_repo_discussions
create_discussion = api.create_discussion
create_pull_request = api.create_pull_request
change_discussion_status = api.change_discussion_status
comment_discussion = api.comment_discussion
edit_discussion_comment = api.edit_discussion_comment
rename_discussion = api.rename_discussion
merge_pull_request = api.merge_pull_request

_validate_or_retrieve_token = api._validate_or_retrieve_token
