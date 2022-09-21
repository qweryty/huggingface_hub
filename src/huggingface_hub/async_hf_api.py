import warnings
from typing import AsyncIterator, BinaryIO, Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import quote

import httpx
from httpx import HTTPError

from ._commit_api import (
    CommitOperation,
    CommitOperationAdd,
    CommitOperationDelete,
    fetch_upload_modes,
    prepare_commit_payload,
    upload_lfs_files,
)
from .community import (
    Discussion,
    DiscussionComment,
    DiscussionStatusChange,
    DiscussionTitleChange,
    DiscussionWithDetails,
    deserialize_event,
)
from .constants import (
    DEFAULT_REVISION,
    ENDPOINT,
    REGEX_COMMIT_OID,
    REPO_TYPE_MODEL,
    REPO_TYPES,
    REPO_TYPES_URL_PREFIXES,
    SPACES_SDK_TYPES,
    USERNAME_PLACEHOLDER,
)
from .hf_api_models import DatasetInfo, HfFolder, MetricInfo, ModelInfo, SpaceInfo
from .hf_api_utils import (
    _parse_revision_from_pr_url,
    _prepare_upload_folder_commit,
    _validate_repo_id_deprecation,
    erase_from_credential_store,
    repo_type_and_id_from_hf_id,
    write_to_credential_store,
)
from .utils import (
    RepositoryNotFoundError,
    hf_raise_for_status,
    logging,
    parse_datetime,
    validate_hf_hub_args,
)
from .utils._deprecation import _deprecate_positional_args
from .utils._typing import Literal
from .utils.endpoint_helpers import (
    DatasetFilter,
    DatasetTags,
    ModelFilter,
    ModelTags,
    _filter_emissions,
)


logger = logging.get_logger(__name__)


class AsyncHfApi:
    def __init__(self, endpoint: Optional[str] = None):
        self.endpoint = endpoint if endpoint is not None else ENDPOINT

    async def whoami(self, token: Optional[str] = None) -> Dict:
        """
        Call HF API to know "whoami".

        Args:
            token (`str`, *optional*):
                Hugging Face token. Will default to the locally saved token if
                not provided.
        """
        if token is None:
            token = HfFolder.get_token()
        if token is None:
            raise ValueError(
                "You need to pass a valid `token` or login by using `huggingface-cli"
                " login`"
            )
        path = f"{self.endpoint}/api/whoami-v2"
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.get(path, headers={"authorization": f"Bearer {token}"})
        try:
            hf_raise_for_status(r)
        except HTTPError as e:
            raise HTTPError(
                "Invalid user token. If you didn't pass a user token, make sure you "
                "are properly logged in by executing `huggingface-cli login`, and "
                "if you did pass a user token, double-check it's correct."
            ) from e
        return r.json()

    async def _is_valid_token(self, token: str):
        """
        Determines whether `token` is a valid token or not.

        Args:
            token (`str`):
                The token to check for validity.

        Returns:
            `bool`: `True` if valid, `False` otherwise.
        """
        try:
            await self.whoami(token=token)
            return True
        except HTTPError:
            return False

    async def _validate_or_retrieve_token(
        self,
        token: Optional[Union[str, bool]] = None,
        name: Optional[str] = None,
        function_name: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Retrieves and validates stored token or validates passed token.
        Args:
            token (``str``, `optional`):
                Hugging Face token. Will default to the locally saved token if not provided.
            name (``str``, `optional`):
                Name of the repository. This is deprecated in favor of repo_id and will be removed in v0.8.
            function_name (``str``, `optional`):
                If _validate_or_retrieve_token is called from a function, name of that function to be passed inside deprecation warning.
        Returns:
            Validated token and the name of the repository.
        Raises:
            [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError)
              If the token is not passed and there's no token saved locally.
            [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              If organization token or invalid token is passed.
        """
        if token is None or token is True:
            token = HfFolder.get_token()
            if token is None:
                raise EnvironmentError(
                    "You need to provide a `token` or be logged in to Hugging Face with"
                    " `huggingface-cli login`."
                )
        if name is not None:
            if await self._is_valid_token(name):
                # TODO(0.6) REMOVE
                warnings.warn(
                    f"`{function_name}` now takes `token` as an optional positional"
                    " argument. Be sure to adapt your code!",
                    FutureWarning,
                )
                token, name = name, token
        if isinstance(token, str):
            if token.startswith("api_org"):
                raise ValueError("You must use your personal account token.")
            if not await self._is_valid_token(token):
                raise ValueError("Invalid token passed!")

        return token, name

    async def _build_auth_headers(
        self, *, token: Optional[str], use_auth_token: Optional[Union[str, bool]]
    ) -> Dict[str, str]:
        """Helper to build Authorization header from kwargs. To be removed in 0.12.0 when `token` is deprecated."""
        if token is not None:
            warnings.warn(
                "`token` is deprecated and will be removed in 0.12.0. Use"
                " `use_auth_token` instead.",
                FutureWarning,
            )

        auth_token = None
        if use_auth_token is None and token is None:
            # To maintain backwards-compatibility. To be removed in 0.12.0
            auth_token = HfFolder.get_token()
        elif use_auth_token:
            auth_token, _ = await self._validate_or_retrieve_token(use_auth_token)
        else:
            auth_token = token
        return {"authorization": f"Bearer {auth_token}"} if auth_token else {}

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

    async def get_model_tags(self) -> ModelTags:
        "Gets all valid model tags as a nested namespace object"
        path = f"{self.endpoint}/api/models-tags-by-type"
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.get(path)
        hf_raise_for_status(r)
        d = r.json()
        return ModelTags(d)

    async def get_dataset_tags(self) -> DatasetTags:
        """
        Gets all valid dataset tags as a nested namespace object.
        """
        path = f"{self.endpoint}/api/datasets-tags-by-type"
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.get(path)
        hf_raise_for_status(r)
        d = r.json()
        return DatasetTags(d)

    async def list_models(
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
        card_data: Optional[bool] = None,
        fetch_config: Optional[bool] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
    ) -> List[ModelInfo]:
        """
        Get the public list of all the models on huggingface.co

        Args:
            filter ([`ModelFilter`] or `str` or `Iterable`, *optional*):
                A string or [`ModelFilter`] which can be used to identify models
                on the Hub.
            author (`str`, *optional*):
                A string which identify the author (user or organization) of the
                returned models
            search (`str`, *optional*):
                A string that will be contained in the returned models Example
                usage:
            emissions_thresholds (`Tuple`, *optional*):
                A tuple of two ints or floats representing a minimum and maximum
                carbon footprint to filter the resulting models with in grams.
            sort (`Literal["lastModified"]` or `str`, *optional*):
                The key with which to sort the resulting models. Possible values
                are the properties of the [`huggingface_hub.hf_api.ModelInfo`] class.
            direction (`Literal[-1]` or `int`, *optional*):
                Direction in which to sort. The value `-1` sorts by descending
                order while all other values sort by ascending order.
            limit (`int`, *optional*):
                The limit on the number of models fetched. Leaving this option
                to `None` fetches all models.
            full (`bool`, *optional*):
                Whether to fetch all model data, including the `lastModified`,
                the `sha`, the files and the `tags`. This is set to `True` by
                default when using a filter.
            card_data (`bool`, *optional*):
                Whether to grab the metadata for the model as well. Can contain
                useful information such as carbon emissions, metrics, and
                datasets trained on.
            fetch_config (`bool`, *optional*):
                Whether to fetch the model configs as well. This is not included
                in `full` due to its size.
            use_auth_token (`bool` or `str`, *optional*):
                Whether to use the `auth_token` provided from the
                `huggingface_hub` cli. If not logged in, a valid `auth_token`
                can be passed in as a string.

        Returns: List of [`huggingface_hub.hf_api.ModelInfo`] objects

        Example usage with the `filter` argument:

        ```python
        >>> from huggingface_hub import HfApi

        >>> api = HfApi()

        >>> # List all models
        >>> api.list_models()

        >>> # Get all valid search arguments
        >>> args = ModelSearchArguments()

        >>> # List only the text classification models
        >>> api.list_models(filter="text-classification")
        >>> # Using the `ModelFilter`
        >>> filt = ModelFilter(task="text-classification")
        >>> # With `ModelSearchArguments`
        >>> filt = ModelFilter(task=args.pipeline_tags.TextClassification)
        >>> api.list_models(filter=filt)

        >>> # Using `ModelFilter` and `ModelSearchArguments` to find text classification in both PyTorch and TensorFlow
        >>> filt = ModelFilter(
        ...     task=args.pipeline_tags.TextClassification,
        ...     library=[args.library.PyTorch, args.library.TensorFlow],
        ... )
        >>> api.list_models(filter=filt)

        >>> # List only models from the AllenNLP library
        >>> api.list_models(filter="allennlp")
        >>> # Using `ModelFilter` and `ModelSearchArguments`
        >>> filt = ModelFilter(library=args.library.allennlp)
        ```

        Example usage with the `search` argument:

        ```python
        >>> from huggingface_hub import HfApi

        >>> api = HfApi()

        >>> # List all models with "bert" in their name
        >>> api.list_models(search="bert")

        >>> # List all models with "bert" in their name made by google
        >>> api.list_models(search="bert", author="google")
        ```
        """
        path = f"{self.endpoint}/api/models"
        headers = {}
        if use_auth_token:
            token, name = await self._validate_or_retrieve_token(use_auth_token)
            headers["authorization"] = f"Bearer {token}"
        params = {}
        if filter is not None:
            if isinstance(filter, ModelFilter):
                params = self._unpack_model_filter(filter)
            else:
                params.update({"filter": filter})
            params.update({"full": True})
        if author is not None:
            params.update({"author": author})
        if search is not None:
            params.update({"search": search})
        if sort is not None:
            params.update({"sort": sort})
        if direction is not None:
            params.update({"direction": direction})
        if limit is not None:
            params.update({"limit": limit})
        if full is not None:
            if full:
                params.update({"full": True})
            elif "full" in params:
                del params["full"]
        if fetch_config is not None:
            params.update({"config": fetch_config})
        if card_data is not None:
            params.update({"cardData": card_data})
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.get(path, params=params, headers=headers)
        hf_raise_for_status(r)
        d = r.json()
        res = [ModelInfo(**x) for x in d]
        if emissions_thresholds is not None:
            if card_data is None:
                raise ValueError(
                    "`emissions_thresholds` were passed without setting"
                    " `card_data=True`."
                )
            else:
                return _filter_emissions(res, *emissions_thresholds)
        return res

    def _unpack_model_filter(self, model_filter: ModelFilter):
        """
        Unpacks a [`ModelFilter`] into something readable for `list_models`
        """
        model_str = ""
        tags = []

        # Handling author
        if model_filter.author is not None:
            model_str = f"{model_filter.author}/"

        # Handling model_name
        if model_filter.model_name is not None:
            model_str += model_filter.model_name

        filter_tuple = []

        # Handling tasks
        if model_filter.task is not None:
            filter_tuple.extend(
                [model_filter.task]
                if isinstance(model_filter.task, str)
                else model_filter.task
            )

        # Handling dataset
        if model_filter.trained_dataset is not None:
            if not isinstance(model_filter.trained_dataset, (list, tuple)):
                model_filter.trained_dataset = [model_filter.trained_dataset]
            for dataset in model_filter.trained_dataset:
                if "dataset:" not in dataset:
                    dataset = f"dataset:{dataset}"
                filter_tuple.append(dataset)

        # Handling library
        if model_filter.library:
            filter_tuple.extend(
                [model_filter.library]
                if isinstance(model_filter.library, str)
                else model_filter.library
            )

        # Handling tags
        if model_filter.tags:
            tags.extend(
                [model_filter.tags]
                if isinstance(model_filter.tags, str)
                else model_filter.tags
            )

        query_dict = {}
        if model_str is not None:
            query_dict["search"] = model_str
        if len(tags) > 0:
            query_dict["tags"] = tags
        if model_filter.language is not None:
            filter_tuple.append(model_filter.language)
        query_dict["filter"] = tuple(filter_tuple)
        return query_dict

    async def list_datasets(
        self,
        *,
        filter: Union[DatasetFilter, str, Iterable[str], None] = None,
        author: Optional[str] = None,
        search: Optional[str] = None,
        sort: Union[Literal["lastModified"], str, None] = None,
        direction: Optional[Literal[-1]] = None,
        limit: Optional[int] = None,
        card_data: Optional[bool] = None,
        full: Optional[bool] = None,
        use_auth_token: Optional[str] = None,
    ) -> List[DatasetInfo]:
        """
        Get the public list of all the datasets on huggingface.co

        Args:
            filter ([`DatasetFilter`] or `str` or `Iterable`, *optional*):
                A string or [`DatasetFilter`] which can be used to identify
                datasets on the hub.
            author (`str`, *optional*):
                A string which identify the author of the returned models
            search (`str`, *optional*):
                A string that will be contained in the returned models.
            sort (`Literal["lastModified"]` or `str`, *optional*):
                The key with which to sort the resulting datasets. Possible
                values are the properties of the [`huggingface_hub.hf_api.DatasetInfo`] class.
            direction (`Literal[-1]` or `int`, *optional*):
                Direction in which to sort. The value `-1` sorts by descending
                order while all other values sort by ascending order.
            limit (`int`, *optional*):
                The limit on the number of datasets fetched. Leaving this option
                to `None` fetches all datasets.
            card_data (`bool`, *optional*):
                Whether to grab the metadata for the dataset as well. Can
                contain useful information such as the PapersWithCode ID.
            full (`bool`, *optional*):
                Whether to fetch all dataset data, including the `lastModified`
                and the `card_data`.
            use_auth_token (`bool` or `str`, *optional*):
                Whether to use the `auth_token` provided from the
                `huggingface_hub` cli. If not logged in, a valid `auth_token`
                can be passed in as a string.

        Example usage with the `filter` argument:

        ```python
        >>> from huggingface_hub import HfApi

        >>> api = HfApi()

        >>> # List all datasets
        >>> api.list_datasets()

        >>> # Get all valid search arguments
        >>> args = DatasetSearchArguments()

        >>> # List only the text classification datasets
        >>> api.list_datasets(filter="task_categories:text-classification")
        >>> # Using the `DatasetFilter`
        >>> filt = DatasetFilter(task_categories="text-classification")
        >>> # With `DatasetSearchArguments`
        >>> filt = DatasetFilter(task=args.task_categories.text_classification)
        >>> api.list_models(filter=filt)

        >>> # List only the datasets in russian for language modeling
        >>> api.list_datasets(
        ...     filter=("languages:ru", "task_ids:language-modeling")
        ... )
        >>> # Using the `DatasetFilter`
        >>> filt = DatasetFilter(languages="ru", task_ids="language-modeling")
        >>> # With `DatasetSearchArguments`
        >>> filt = DatasetFilter(
        ...     languages=args.languages.ru,
        ...     task_ids=args.task_ids.language_modeling,
        ... )
        >>> api.list_datasets(filter=filt)
        ```

        Example usage with the `search` argument:

        ```python
        >>> from huggingface_hub import HfApi

        >>> api = HfApi()

        >>> # List all datasets with "text" in their name
        >>> api.list_datasets(search="text")

        >>> # List all datasets with "text" in their name made by google
        >>> api.list_datasets(search="text", author="google")
        ```
        """
        path = f"{self.endpoint}/api/datasets"
        headers = {}
        if use_auth_token:
            token, name = await self._validate_or_retrieve_token(use_auth_token)
            headers["authorization"] = f"Bearer {token}"
        params = {}
        if filter is not None:
            if isinstance(filter, DatasetFilter):
                params = self._unpack_dataset_filter(filter)
            else:
                params.update({"filter": filter})
        if author is not None:
            params.update({"author": author})
        if search is not None:
            params.update({"search": search})
        if sort is not None:
            params.update({"sort": sort})
        if direction is not None:
            params.update({"direction": direction})
        if limit is not None:
            params.update({"limit": limit})
        if full is not None:
            if full:
                params.update({"full": True})
        if card_data is not None:
            if card_data:
                params.update({"full": True})
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.get(path, params=params, headers=headers)
        hf_raise_for_status(r)
        d = r.json()
        return [DatasetInfo(**x) for x in d]

    def _unpack_dataset_filter(self, dataset_filter: DatasetFilter):
        """
        Unpacks a [`DatasetFilter`] into something readable for `list_datasets`
        """
        dataset_str = ""

        # Handling author
        if dataset_filter.author is not None:
            dataset_str = f"{dataset_filter.author}/"

        # Handling dataset_name
        if dataset_filter.dataset_name is not None:
            dataset_str += dataset_filter.dataset_name

        filter_tuple = []
        data_attributes = [
            "benchmark",
            "language_creators",
            "languages",
            "multilinguality",
            "size_categories",
            "task_categories",
            "task_ids",
        ]

        for attr in data_attributes:
            curr_attr = getattr(dataset_filter, attr)
            if curr_attr is not None:
                if not isinstance(curr_attr, (list, tuple)):
                    curr_attr = [curr_attr]
                for data in curr_attr:
                    if f"{attr}:" not in data:
                        data = f"{attr}:{data}"
                    filter_tuple.append(data)

        query_dict = {}
        if dataset_str is not None:
            query_dict["search"] = dataset_str
        query_dict["filter"] = tuple(filter_tuple)
        return query_dict

    async def list_metrics(self) -> List[MetricInfo]:
        """
        Get the public list of all the metrics on huggingface.co

        Returns:
            `List[MetricInfo]`: a list of [`MetricInfo`] objects which.
        """
        path = f"{self.endpoint}/api/metrics"
        params = {}
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.get(path, params=params)
        hf_raise_for_status(r)
        d = r.json()
        return [MetricInfo(**x) for x in d]

    async def list_spaces(
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
        """
        Get the public list of all Spaces on huggingface.co

        Args:
            filter `str` or `Iterable`, *optional*):
                A string tag or list of tags that can be used to identify Spaces on the Hub.
            author (`str`, *optional*):
                A string which identify the author of the returned Spaces.
            search (`str`, *optional*):
                A string that will be contained in the returned Spaces.
            sort (`Literal["lastModified"]` or `str`, *optional*):
                The key with which to sort the resulting Spaces. Possible
                values are the properties of the [`huggingface_hub.hf_api.SpaceInfo`]` class.
            direction (`Literal[-1]` or `int`, *optional*):
                Direction in which to sort. The value `-1` sorts by descending
                order while all other values sort by ascending order.
            limit (`int`, *optional*):
                The limit on the number of Spaces fetched. Leaving this option
                to `None` fetches all Spaces.
            datasets (`str` or `Iterable`, *optional*):
                Whether to return Spaces that make use of a dataset.
                The name of a specific dataset can be passed as a string.
            models (`str` or `Iterable`, *optional*):
                Whether to return Spaces that make use of a model.
                The name of a specific model can be passed as a string.
            linked (`bool`, *optional*):
                Whether to return Spaces that make use of either a model or a dataset.
            full (`bool`, *optional*):
                Whether to fetch all Spaces data, including the `lastModified`
                and the `card_data`.
            use_auth_token (`bool` or `str`, *optional*):
                Whether to use the `auth_token` provided from the
                `huggingface_hub` cli. If not logged in, a valid `auth_token`
                can be passed in as a string.

        Returns:
            `List[SpaceInfo]`: a list of [`huggingface_hub.hf_api.SpaceInfo`] objects
        """
        path = f"{self.endpoint}/api/spaces"
        headers = {}
        if use_auth_token:
            token, name = await self._validate_or_retrieve_token(use_auth_token)
            headers["authorization"] = f"Bearer {token}"
        params = {}
        if filter is not None:
            params.update({"filter": filter})
        if author is not None:
            params.update({"author": author})
        if search is not None:
            params.update({"search": search})
        if sort is not None:
            params.update({"sort": sort})
        if direction is not None:
            params.update({"direction": direction})
        if limit is not None:
            params.update({"limit": limit})
        if full is not None:
            if full:
                params.update({"full": True})
        if linked is not None:
            if linked:
                params.update({"linked": True})
        if datasets is not None:
            params.update({"datasets": datasets})
        if models is not None:
            params.update({"models": models})
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.get(path, params=params, headers=headers)
        hf_raise_for_status(r)
        d = r.json()
        return [SpaceInfo(**x) for x in d]

    @validate_hf_hub_args
    async def model_info(
        self,
        repo_id: str,
        *,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
        security_status: Optional[bool] = None,
        files_metadata: bool = False,
        use_auth_token: Optional[Union[bool, str]] = None,
    ) -> ModelInfo:
        """
        Get info on one specific model on huggingface.co

        Model can be private if you pass an acceptable token or are logged in.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            revision (`str`, *optional*):
                The revision of the model repository from which to get the
                information.
            token (`str`, *optional*):
                Deprecated in favor of `use_auth_token`. Will be removed in 0.12.0.
                An authentication token (See https://huggingface.co/settings/token)
            timeout (`float`, *optional*):
                Whether to set a timeout for the request to the Hub.
            security_status (`bool`, *optional*):
                Whether to retrieve the security status from the model
                repository as well.
            files_metadata (`bool`, *optional*):
                Whether or not to retrieve metadata for files in the repository
                (size, LFS metadata, etc). Defaults to `False`.
            use_auth_token (`bool` or `str`, *optional*):
                Whether to use the `auth_token` provided from the
                `huggingface_hub` cli. If not logged in, a valid `auth_token`
                can be passed in as a string.

        Returns:
            [`huggingface_hub.hf_api.ModelInfo`]: The model repository information.

        <Tip>

        Raises the following errors:

            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.
            - [`~utils.RevisionNotFoundError`]
              If the revision to download from cannot be found.

        </Tip>
        """
        headers = await self._build_auth_headers(
            token=token, use_auth_token=use_auth_token
        )
        path = (
            f"{self.endpoint}/api/models/{repo_id}"
            if revision is None
            else (
                f"{self.endpoint}/api/models/{repo_id}/revision/{quote(revision, safe='')}"
            )
        )
        params = {}
        if security_status:
            params["securityStatus"] = True
        if files_metadata:
            params["blobs"] = True
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.get(
                path,
                headers=headers,
                timeout=timeout,
                params=params,
            )
        hf_raise_for_status(r)
        d = r.json()
        return ModelInfo(**d)

    @validate_hf_hub_args
    async def dataset_info(
        self,
        repo_id: str,
        *,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
        files_metadata: bool = False,
        use_auth_token: Optional[Union[bool, str]] = None,
    ) -> DatasetInfo:
        """
        Get info on one specific dataset on huggingface.co.

        Dataset can be private if you pass an acceptable token.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            revision (`str`, *optional*):
                The revision of the dataset repository from which to get the
                information.
            token (`str`, *optional*):
                Deprecated in favor of `use_auth_token`. Will be removed in 0.12.0.
                An authentication token (See https://huggingface.co/settings/token)
            timeout (`float`, *optional*):
                Whether to set a timeout for the request to the Hub.
            files_metadata (`bool`, *optional*):
                Whether or not to retrieve metadata for files in the repository
                (size, LFS metadata, etc). Defaults to `False`.
            use_auth_token (`bool` or `str`, *optional*):
                Whether to use the `auth_token` provided from the
                `huggingface_hub` cli. If not logged in, a valid `auth_token`
                can be passed in as a string.

        Returns:
            [`hf_api.DatasetInfo`]: The dataset repository information.

        <Tip>

        Raises the following errors:

            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.
            - [`~utils.RevisionNotFoundError`]
              If the revision to download from cannot be found.

        </Tip>
        """
        headers = await self._build_auth_headers(
            token=token, use_auth_token=use_auth_token
        )

        path = (
            f"{self.endpoint}/api/datasets/{repo_id}"
            if revision is None
            else (
                f"{self.endpoint}/api/datasets/{repo_id}/revision/{quote(revision, safe='')}"
            )
        )
        params = {}
        if files_metadata:
            params["blobs"] = True

        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.get(path, headers=headers, timeout=timeout, params=params)
        hf_raise_for_status(r)
        d = r.json()
        return DatasetInfo(**d)

    @validate_hf_hub_args
    async def space_info(
        self,
        repo_id: str,
        *,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
        files_metadata: bool = False,
        use_auth_token: Optional[Union[bool, str]] = None,
    ) -> SpaceInfo:
        """
        Get info on one specific Space on huggingface.co.

        Space can be private if you pass an acceptable token.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            revision (`str`, *optional*):
                The revision of the space repository from which to get the
                information.
            token (`str`, *optional*):
                Deprecated in favor of `use_auth_token`. Will be removed in 0.12.0.
                An authentication token (See https://huggingface.co/settings/token)
            timeout (`float`, *optional*):
                Whether to set a timeout for the request to the Hub.
            files_metadata (`bool`, *optional*):
                Whether or not to retrieve metadata for files in the repository
                (size, LFS metadata, etc). Defaults to `False`.
            use_auth_token (`bool` or `str`, *optional*):
                Whether to use the `auth_token` provided from the
                `huggingface_hub` cli. If not logged in, a valid `auth_token`
                can be passed in as a string.

        Returns:
            [`~hf_api.SpaceInfo`]: The space repository information.

        <Tip>

        Raises the following errors:

            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.
            - [`~utils.RevisionNotFoundError`]
              If the revision to download from cannot be found.

        </Tip>
        """
        headers = await self._build_auth_headers(
            token=token, use_auth_token=use_auth_token
        )
        path = (
            f"{self.endpoint}/api/spaces/{repo_id}"
            if revision is None
            else (
                f"{self.endpoint}/api/spaces/{repo_id}/revision/{quote(revision, safe='')}"
            )
        )
        params = {}
        if files_metadata:
            params["blobs"] = True

        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.get(path, headers=headers, timeout=timeout, params=params)
        hf_raise_for_status(r)
        d = r.json()
        return SpaceInfo(**d)

    @validate_hf_hub_args
    async def repo_info(
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
        """
        Get the info object for a given repo of a given type.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            revision (`str`, *optional*):
                The revision of the repository from which to get the
                information.
            token (`str`, *optional*):
                Deprecated in favor of `use_auth_token`. Will be removed in 0.12.0.
                An authentication token (See https://huggingface.co/settings/token)
            timeout (`float`, *optional*):
                Whether to set a timeout for the request to the Hub.
            files_metadata (`bool`, *optional*):
                Whether or not to retrieve metadata for files in the repository
                (size, LFS metadata, etc). Defaults to `False`.
            use_auth_token (`bool` or `str`, *optional*):
                Whether to use the `auth_token` provided from the
                `huggingface_hub` cli. If not logged in, a valid `auth_token`
                can be passed in as a string.

        Returns:
            `Union[SpaceInfo, DatasetInfo, ModelInfo]`: The repository information, as a
            [`huggingface_hub.hf_api.DatasetInfo`], [`huggingface_hub.hf_api.ModelInfo`]
            or [`huggingface_hub.hf_api.SpaceInfo`] object.

        <Tip>

        Raises the following errors:

            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.
            - [`~utils.RevisionNotFoundError`]
              If the revision to download from cannot be found.

        </Tip>
        """
        if repo_type is None or repo_type == "model":
            method = self.model_info
        elif repo_type == "dataset":
            method = self.dataset_info
        elif repo_type == "space":
            method = self.space_info
        else:
            raise ValueError("Unsupported repo type.")
        return await method(
            repo_id,
            revision=revision,
            token=token,
            timeout=timeout,
            files_metadata=files_metadata,
            use_auth_token=use_auth_token,
        )

    @validate_hf_hub_args
    async def list_repo_files(
        self,
        repo_id: str,
        *,
        revision: Optional[str] = None,
        repo_type: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
    ) -> List[str]:
        """
        Get the list of files in a given repo.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            revision (`str`, *optional*):
                The revision of the model repository from which to get the
                information.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                Deprecated in favor of `use_auth_token`. Will be removed in 0.12.0.
                An authentication token (See https://huggingface.co/settings/token)
            timeout (`float`, *optional*):
                Whether to set a timeout for the request to the Hub.
            use_auth_token (`bool` or `str`, *optional*):
                Whether to use the `auth_token` provided from the
                `huggingface_hub` cli. If not logged in, a valid `auth_token`
                can be passed in as a string.

        Returns:
            `List[str]`: the list of files in a given repository.
        """
        repo_info = await self.repo_info(
            repo_id,
            revision=revision,
            repo_type=repo_type,
            token=token,
            timeout=timeout,
            use_auth_token=use_auth_token,
        )
        return [f.rfilename for f in repo_info.siblings]

    @validate_hf_hub_args
    @_deprecate_positional_args(version="0.12")
    async def create_repo(
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
        """Create an empty repo on the HuggingFace Hub.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.

                <Tip>

                Version added: 0.5

                </Tip>

            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)
            private (`bool`, *optional*, defaults to `False`):
                Whether the model repo should be private.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            exist_ok (`bool`, *optional*, defaults to `False`):
                If `True`, do not raise an error if repo already exists.
            space_sdk (`str`, *optional*):
                Choice of SDK to use if repo_type is "space". Can be
                "streamlit", "gradio", or "static".

        Returns:
            `str`: URL to the newly created repo.
        """
        name, organization = _validate_repo_id_deprecation(repo_id, name, organization)

        path = f"{self.endpoint}/api/repos/create"

        token, name = await self._validate_or_retrieve_token(
            token, name, function_name="create_repo"
        )

        checked_name = repo_type_and_id_from_hf_id(name)

        if (
            repo_type is not None
            and checked_name[0] is not None
            and repo_type != checked_name[0]
        ):
            raise ValueError(
                f"""Passed `repo_type` and found `repo_type` are not the same ({repo_type},
{checked_name[0]}).
                Please make sure you are expecting the right type of repository to
                exist."""
            )

        if (
            organization is not None
            and checked_name[1] is not None
            and organization != checked_name[1]
        ):
            raise ValueError(
                f"""Passed `organization` and `name` organization are not the same ({organization},
{checked_name[1]}).
                Please either include the organization in only `name` or the
                `organization` parameter, such as
                `api.create_repo({checked_name[0]}, organization={organization})` or
                `api.create_repo({checked_name[1]}/{checked_name[2]})`"""
            )

        repo_type = repo_type or checked_name[0]
        organization = organization or checked_name[1]
        name = checked_name[2]

        if repo_type not in REPO_TYPES:
            raise ValueError("Invalid repo type")

        json = {"name": name, "organization": organization, "private": private}
        if repo_type is not None:
            json["type"] = repo_type
        if repo_type == "space":
            if space_sdk is None:
                raise ValueError(
                    "No space_sdk provided. `create_repo` expects space_sdk to be one"
                    f" of {SPACES_SDK_TYPES} when repo_type is 'space'`"
                )
            if space_sdk not in SPACES_SDK_TYPES:
                raise ValueError(
                    f"Invalid space_sdk. Please choose one of {SPACES_SDK_TYPES}."
                )
            json["sdk"] = space_sdk
        if space_sdk is not None and repo_type != "space":
            warnings.warn(
                "Ignoring provided space_sdk because repo_type is not 'space'."
            )

        if getattr(self, "_lfsmultipartthresh", None):
            json["lfsmultipartthresh"] = self._lfsmultipartthresh
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.post(
                path,
                headers={"authorization": f"Bearer {token}"},
                json=json,
            )

        try:
            hf_raise_for_status(r)
        except HTTPError as err:
            if not (exist_ok and err.response.status_code == 409):
                try:
                    additional_info = r.json().get("error", None)
                    if additional_info:
                        new_err = f"{err.args[0]} - {additional_info}"
                        err.args = (new_err,) + err.args[1:]
                except ValueError:
                    pass

                raise err

        d = r.json()
        return d["url"]

    @validate_hf_hub_args
    async def delete_repo(
        self,
        repo_id: str = None,
        *,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ):
        """
        Delete a repo from the HuggingFace Hub. CAUTION: this is irreversible.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.

                <Tip>

                Version added: 0.5

                </Tip>

            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model.

        <Tip>

        Raises the following errors:

            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        organization, name = repo_id.split("/") if "/" in repo_id else (None, repo_id)

        path = f"{self.endpoint}/api/repos/delete"

        token, name = await self._validate_or_retrieve_token(
            token, name, function_name="delete_repo"
        )

        checked_name = repo_type_and_id_from_hf_id(name)

        if (
            repo_type is not None
            and checked_name[0] is not None
            and repo_type != checked_name[0]
        ):
            raise ValueError(
                f"""Passed `repo_type` and found `repo_type` are not the same ({repo_type},
{checked_name[0]}).
                Please make sure you are expecting the right type of repository to
                exist."""
            )

        if (
            organization is not None
            and checked_name[1] is not None
            and organization != checked_name[1]
        ):
            raise ValueError(
                "Passed `organization` and `name` organization are not the same"
                f" ({organization}, {checked_name[1]})."
                "\nPlease either include the organization in only `name` or the"
                " `organization` parameter, such as "
                f"`api.create_repo({checked_name[0]}, organization={organization})` "
                f"or `api.create_repo({checked_name[1]}/{checked_name[2]})`"
            )

        repo_type = repo_type or checked_name[0]
        organization = organization or checked_name[1]
        name = checked_name[2]

        if repo_type not in REPO_TYPES:
            raise ValueError("Invalid repo type")

        json = {"name": name, "organization": organization}
        if repo_type is not None:
            json["type"] = repo_type

        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.request(
                method="DELETE",
                url=path,
                headers={"authorization": f"Bearer {token}"},
                json=json,
            )
        hf_raise_for_status(r)

    @validate_hf_hub_args
    async def update_repo_visibility(
        self,
        repo_id: str = None,
        private: bool = False,
        *,
        token: Optional[str] = None,
        organization: Optional[str] = None,
        repo_type: Optional[str] = None,
        name: str = None,
    ) -> Dict[str, bool]:
        """Update the visibility setting of a repository.

        Args:
            repo_id (`str`, *optional*):
                A namespace (user or an organization) and a repo name separated
                by a `/`.

                <Tip>

                Version added: 0.5

                </Tip>

            private (`bool`, *optional*, defaults to `False`):
                Whether the model repo should be private.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.

        Returns:
            The HTTP response in json.

        <Tip>

        Raises the following errors:

            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        if repo_type not in REPO_TYPES:
            raise ValueError("Invalid repo type")

        organization, name = repo_id.split("/") if "/" in repo_id else (None, repo_id)

        token, name = await self._validate_or_retrieve_token(
            token, name, function_name="update_repo_visibility"
        )

        if organization is None:
            namespace = (await self.whoami(token))["name"]
        else:
            namespace = organization

        if repo_type is None:
            repo_type = REPO_TYPE_MODEL  # default repo type

        path = f"{self.endpoint}/api/{repo_type}s/{namespace}/{name}/settings"

        json = {"private": private}

        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.put(
                path,
                headers={"authorization": f"Bearer {token}"},
                json=json,
            )
        hf_raise_for_status(r)
        return r.json()

    async def move_repo(
        self,
        from_id: str,
        to_id: str,
        *,
        repo_type: Optional[str] = None,
        token: Optional[str] = None,
    ):
        """
        Moving a repository from namespace1/repo_name1 to namespace2/repo_name2

        Note there are certain limitations. For more information about moving
        repositories, please see
        https://hf.co/docs/hub/main#how-can-i-rename-or-transfer-a-repo.

        Args:
            from_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`. Original repository identifier.
            to_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`. Final repository identifier.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)

        <Tip>

        Raises the following errors:

            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """

        token, name = await self._validate_or_retrieve_token(token)

        if len(from_id.split("/")) != 2:
            raise ValueError(
                f"Invalid repo_id: {from_id}. It should have a namespace"
                " (:namespace:/:repo_name:)"
            )

        if len(to_id.split("/")) != 2:
            raise ValueError(
                f"Invalid repo_id: {to_id}. It should have a namespace"
                " (:namespace:/:repo_name:)"
            )

        json = {"fromRepo": from_id, "toRepo": to_id, "type": repo_type}

        path = f"{self.endpoint}/api/repos/move"
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.post(
                path,
                headers={"authorization": f"Bearer {token}"},
                json=json,
            )
        try:
            hf_raise_for_status(r)
        except HTTPError as e:
            if r.text:
                raise HTTPError(
                    f"{r.status_code} Error Message: {r.text}. For additional"
                    " documentation please see"
                    " https://hf.co/docs/hub/main#how-can-i-rename-or-transfer-a-repo."
                ) from e
            else:
                raise e
        logger.info(
            "Accepted transfer request. You will get an email once this is successfully"
            " completed."
        )

    @validate_hf_hub_args
    async def create_commit(
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
        """
        Creates a commit in the given repo, deleting & uploading files as needed.

        Args:
            repo_id (`str`):
                The repository in which the commit will be created, for example:
                `"username/custom_transformers"`

            operations (`Iterable` of [`~hf_api.CommitOperation`]):
                An iterable of operations to include in the commit, either:

                    - [`~hf_api.CommitOperationAdd`] to upload a file
                    - [`~hf_api.CommitOperationDelete`] to delete a file

            commit_message (`str`):
                The summary (first line) of the commit that will be created.

            commit_description (`str`, *optional*):
                The description of the commit that will be created

            token (`str`, *optional*):
                Authentication token, obtained with `HfApi.login` method. Will
                default to the stored token.

            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.

            revision (`str`, *optional*):
                The git revision to commit from. Defaults to the head of the
                `"main"` branch.

            create_pr (`boolean`, *optional*):
                Whether or not to create a Pull Request from `revision` with that commit.
                Defaults to `False`. If set to `True`, this function will return the URL
                to the newly created Pull Request on the Hub.

            num_threads (`int`, *optional*):
                Number of concurrent threads for uploading files. Defaults to 5.
                Setting it to 2 means at most 2 files will be uploaded concurrently.

            parent_commit (`str`, *optional*):
                The OID / SHA of the parent commit, as a hexadecimal string.
                Shorthands (7 first characters) are also supported.If specified and `create_pr` is `False`,
                the commit will fail if `revision` does not point to `parent_commit`. If specified and `create_pr`
                is `True`, the pull request will be created from `parent_commit`. Specifying `parent_commit`
                ensures the repo has not changed before committing the changes, and can be especially useful
                if the repo is updated / committed to concurrently.

        Returns:
            `str` or `None`:
                If `create_pr` is `True`, returns the URL to the newly created Pull Request
                on the Hub. Otherwise returns `None`.

        Raises:
            [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
                If commit message is empty.
            [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
                If parent commit is not a valid commit OID.
            [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
                If the Hub API returns an HTTP 400 error (bad request)
            [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
                If `create_pr` is `True` and revision is neither `None` nor `"main"`.
            [`~utils.RepositoryNotFoundError`]:
                If repository is not found (error 404): wrong repo_id/repo_type, private
                but not authenticated or repo does not exist.

        <Tip warning={true}>

        `create_commit` assumes that the repo already exists on the Hub. If you get a
        Client error 404, please make sure you are authenticated and that `repo_id` and
        `repo_type` are set correctly. If repo does not exist, create it first using
        [`~hf_api.create_repo`].

        </Tip>
        """
        if parent_commit is not None and not REGEX_COMMIT_OID.fullmatch(parent_commit):
            raise ValueError(
                "`parent_commit` is not a valid commit OID. It must match the"
                f" following regex: {REGEX_COMMIT_OID}"
            )

        if commit_message is None or len(commit_message) == 0:
            raise ValueError("`commit_message` can't be empty, please pass a value.")

        commit_description = (
            commit_description if commit_description is not None else ""
        )
        repo_type = repo_type if repo_type is not None else REPO_TYPE_MODEL
        if repo_type not in REPO_TYPES:
            raise ValueError(f"Invalid repo type, must be one of {REPO_TYPES}")
        token, name = await self._validate_or_retrieve_token(token)
        revision = (
            quote(revision, safe="") if revision is not None else DEFAULT_REVISION
        )
        create_pr = create_pr if create_pr is not None else False

        if create_pr and revision != DEFAULT_REVISION:
            raise ValueError("Can only create pull requests against {DEFAULT_REVISION}")

        operations = list(operations)
        additions = [op for op in operations if isinstance(op, CommitOperationAdd)]
        deletions = [op for op in operations if isinstance(op, CommitOperationDelete)]

        if len(additions) + len(deletions) != len(operations):
            raise ValueError(
                "Unknown operation, must be one of `CommitOperationAdd` or"
                " `CommitOperationDelete`"
            )

        logger.debug(
            f"About to commit to the hub: {len(additions)} addition(s) and"
            f" {len(deletions)} deletion(s)."
        )

        for addition in additions:
            addition.validate()

        try:
            additions_with_upload_mode = fetch_upload_modes(
                additions=additions,
                repo_type=repo_type,
                repo_id=repo_id,
                token=token,
                revision=revision,
                endpoint=self.endpoint,
                create_pr=create_pr,
            )
        except RepositoryNotFoundError as e:
            e.append_to_message(
                "\nNote: Creating a commit assumes that the repo already exists on the"
                " Huggingface Hub. Please use `create_repo` if it's not the case."
            )
            raise

        upload_lfs_files(
            additions=[
                addition
                for (addition, upload_mode) in additions_with_upload_mode
                if upload_mode == "lfs"
            ],
            repo_type=repo_type,
            repo_id=repo_id,
            token=token,
            endpoint=self.endpoint,
            num_threads=num_threads,
        )
        commit_payload = prepare_commit_payload(
            additions=additions_with_upload_mode,
            deletions=deletions,
            commit_message=commit_message,
            commit_description=commit_description,
            parent_commit=parent_commit,
        )
        commit_url = f"{self.endpoint}/api/{repo_type}s/{repo_id}/commit/{revision}"

        async with httpx.AsyncClient(follow_redirects=True) as client:
            commit_resp = await client.post(
                url=commit_url,
                headers={"Authorization": f"Bearer {token}"},
                json=commit_payload,
                params={"create_pr": "1"} if create_pr else None,
            )
        hf_raise_for_status(commit_resp, endpoint_name="commit")
        return commit_resp.json().get("pullRequestUrl", None)

    @validate_hf_hub_args
    async def upload_file(
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
        """
        Upload a local file (up to 50 GB) to the given repo. The upload is done
        through a HTTP post request, and doesn't require git or git-lfs to be
        installed.

        Args:
            path_or_fileobj (`str`, `bytes`, or `IO`):
                Path to a file on the local machine or binary data stream /
                fileobj / buffer.
            path_in_repo (`str`):
                Relative filepath in the repo, for example:
                `"checkpoints/1fec34a/weights.bin"`
            repo_id (`str`):
                The repository to which the file will be uploaded, for example:
                `"username/custom_transformers"`
            token (`str`, *optional*):
                Authentication token, obtained with `HfApi.login` method. Will
                default to the stored token.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            revision (`str`, *optional*):
                The git revision to commit from. Defaults to the head of the
                `"main"` branch.
            identical_ok (`bool`, *optional*, defaults to `True`):
                Deprecated: will be removed in 0.11.0.
                Changing this value has no effect.
            commit_message (`str`, *optional*):
                The summary / title / first line of the generated commit
            commit_description (`str` *optional*)
                The description of the generated commit
            create_pr (`boolean`, *optional*):
                Whether or not to create a Pull Request from `revision` with that commit.
                Defaults to `False`.
            parent_commit (`str`, *optional*):
                The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported.
                If specified and `create_pr` is `False`, the commit will fail if `revision` does not point to `parent_commit`.
                If specified and `create_pr` is `True`, the pull request will be created from `parent_commit`.
                Specifying `parent_commit` ensures the repo has not changed before committing the changes, and can be
                especially useful if the repo is updated / committed to concurrently.


        Returns:
            `str`: The URL to visualize the uploaded file on the hub

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://2.python-requests.org/en/master/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.
            - [`~utils.RevisionNotFoundError`]
              If the revision to download from cannot be found.

        </Tip>

        <Tip warning={true}>

        `upload_file` assumes that the repo already exists on the Hub. If you get a
        Client error 404, please make sure you are authenticated and that `repo_id` and
        `repo_type` are set correctly. If repo does not exist, create it first using
        [`~hf_api.create_repo`].

        </Tip>

        Example:

        ```python
        >>> from huggingface_hub import upload_file

        >>> with open("./local/filepath", "rb") as fobj:
        ...     upload_file(
        ...         path_or_fileobj=fileobj,
        ...         path_in_repo="remote/file/path.h5",
        ...         repo_id="username/my-dataset",
        ...         repo_type="datasets",
        ...         token="my_token",
        ...     )
        "https://huggingface.co/datasets/username/my-dataset/blob/main/remote/file/path.h5"

        >>> upload_file(
        ...     path_or_fileobj=".\\\\local\\\\file\\\\path",
        ...     path_in_repo="remote/file/path.h5",
        ...     repo_id="username/my-model",
        ...     token="my_token",
        ... )
        "https://huggingface.co/username/my-model/blob/main/remote/file/path.h5"

        >>> upload_file(
        ...     path_or_fileobj=".\\\\local\\\\file\\\\path",
        ...     path_in_repo="remote/file/path.h5",
        ...     repo_id="username/my-model",
        ...     token="my_token",
        ...     create_pr=True,
        ... )
        "https://huggingface.co/username/my-model/blob/refs%2Fpr%2F1/remote/file/path.h5"
        ```
        """
        if identical_ok is not None:
            warnings.warn(
                "`identical_ok` has no effect and is deprecated. It will be removed in"
                " 0.11.0.",
                FutureWarning,
            )

        if repo_type not in REPO_TYPES:
            raise ValueError(f"Invalid repo type, must be one of {REPO_TYPES}")

        commit_message = (
            commit_message
            if commit_message is not None
            else f"Upload {path_in_repo} with huggingface_hub"
        )
        operation = CommitOperationAdd(
            path_or_fileobj=path_or_fileobj,
            path_in_repo=path_in_repo,
        )

        pr_url = await self.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=[operation],
            commit_message=commit_message,
            commit_description=commit_description,
            token=token,
            revision=revision,
            create_pr=create_pr,
            parent_commit=parent_commit,
        )

        if pr_url is not None:
            revision = quote(_parse_revision_from_pr_url(pr_url), safe="")
        if repo_type in REPO_TYPES_URL_PREFIXES:
            repo_id = REPO_TYPES_URL_PREFIXES[repo_type] + repo_id
        revision = revision if revision is not None else DEFAULT_REVISION
        # Similar to `hf_hub_url` but it's "blob" instead of "resolve"
        return f"{self.endpoint}/{repo_id}/blob/{revision}/{path_in_repo}"

    @validate_hf_hub_args
    async def upload_folder(
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
        """
        Upload a local folder to the given repo. The upload is done
        through a HTTP requests, and doesn't require git or git-lfs to be
        installed.

        The structure of the folder will be preserved. Files with the same name
        already present in the repository will be overwritten, others will be left untouched.

        Use the `allow_patterns` and `ignore_patterns` arguments to specify which files
        to upload. These parameters accept either a single pattern or a list of
        patterns. Patterns are Standard Wildcards (globbing patterns) as documented
        [here](https://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm). If both
        `allow_patterns` and `ignore_patterns` are provided, both constraints apply. By
        default, all files from the folder are uploaded.

        Uses `HfApi.create_commit` under the hood.

        Args:
            repo_id (`str`):
                The repository to which the file will be uploaded, for example:
                `"username/custom_transformers"`
            folder_path (`str`):
                Path to the folder to upload on the local file system
            path_in_repo (`str`, *optional*):
                Relative path of the directory in the repo, for example:
                `"checkpoints/1fec34a/results"`. Will default to the root folder of the repository.
            token (`str`, *optional*):
                Authentication token, obtained with `HfApi.login` method. Will
                default to the stored token.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            revision (`str`, *optional*):
                The git revision to commit from. Defaults to the head of the
                `"main"` branch.
            commit_message (`str`, *optional*):
                The summary / title / first line of the generated commit. Defaults to:
                `f"Upload {path_in_repo} with huggingface_hub"`
            commit_description (`str` *optional*):
                The description of the generated commit
            create_pr (`boolean`, *optional*):
                Whether or not to create a Pull Request from the pushed changes. Defaults
                to `False`.
            parent_commit (`str`, *optional*):
                The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported.
                If specified and `create_pr` is `False`, the commit will fail if `revision` does not point to `parent_commit`.
                If specified and `create_pr` is `True`, the pull request will be created from `parent_commit`.
                Specifying `parent_commit` ensures the repo has not changed before committing the changes, and can be
                especially useful if the repo is updated / committed to concurrently.
            allow_patterns (`List[str]` or `str`, *optional*):
                If provided, only files matching at least one pattern are uploaded.
            ignore_patterns (`List[str]` or `str`, *optional*):
                If provided, files matching any of the patterns are not uploaded.

        Returns:
            `str`: A URL to visualize the uploaded folder on the hub

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://2.python-requests.org/en/master/api/#requests.HTTPError)
            if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            if some parameter value is invalid

        </Tip>

        <Tip warning={true}>

        `upload_folder` assumes that the repo already exists on the Hub. If you get a
        Client error 404, please make sure you are authenticated and that `repo_id` and
        `repo_type` are set correctly. If repo does not exist, create it first using
        [`~hf_api.create_repo`].

        </Tip>

        Example:

        ```python
        >>> upload_folder(
        ...     folder_path="local/checkpoints",
        ...     path_in_repo="remote/experiment/checkpoints",
        ...     repo_id="username/my-dataset",
        ...     repo_type="datasets",
        ...     token="my_token",
        ...     ignore_patterns="**/logs/*.txt",
        ... )
        # "https://huggingface.co/datasets/username/my-dataset/tree/main/remote/experiment/checkpoints"

        >>> upload_folder(
        ...     folder_path="local/checkpoints",
        ...     path_in_repo="remote/experiment/checkpoints",
        ...     repo_id="username/my-dataset",
        ...     repo_type="datasets",
        ...     token="my_token",
        ...     create_pr=True,
        ... )
        # "https://huggingface.co/datasets/username/my-dataset/tree/refs%2Fpr%2F1/remote/experiment/checkpoints"

        ```
        """
        if repo_type not in REPO_TYPES:
            raise ValueError(f"Invalid repo type, must be one of {REPO_TYPES}")

        # By default, upload folder to the root directory in repo.
        if path_in_repo is None:
            path_in_repo = ""

        commit_message = (
            commit_message
            if commit_message is not None
            else f"Upload {path_in_repo} with huggingface_hub"
        )

        files_to_add = _prepare_upload_folder_commit(
            folder_path,
            path_in_repo,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

        pr_url = await self.create_commit(
            repo_type=repo_type,
            repo_id=repo_id,
            operations=files_to_add,
            commit_message=commit_message,
            commit_description=commit_description,
            token=token,
            revision=revision,
            create_pr=create_pr,
            parent_commit=parent_commit,
        )

        if pr_url is not None:
            revision = quote(_parse_revision_from_pr_url(pr_url), safe="")
        if repo_type in REPO_TYPES_URL_PREFIXES:
            repo_id = REPO_TYPES_URL_PREFIXES[repo_type] + repo_id
        revision = revision if revision is not None else DEFAULT_REVISION
        # Similar to `hf_hub_url` but it's "tree" instead of "resolve"
        return f"{self.endpoint}/{repo_id}/tree/{revision}/{path_in_repo}"

    @validate_hf_hub_args
    async def delete_file(
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
        """
        Deletes a file in the given repo.

        Args:
            path_in_repo (`str`):
                Relative filepath in the repo, for example:
                `"checkpoints/1fec34a/weights.bin"`
            repo_id (`str`):
                The repository from which the file will be deleted, for example:
                `"username/custom_transformers"`
            token (`str`, *optional*):
                Authentication token, obtained with `HfApi.login` method. Will
                default to the stored token.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if the file is in a dataset or
                space, `None` or `"model"` if in a model. Default is `None`.
            revision (`str`, *optional*):
                The git revision to commit from. Defaults to the head of the
                `"main"` branch.
            commit_message (`str`, *optional*):
                The summary / title / first line of the generated commit. Defaults to
                `f"Delete {path_in_repo} with huggingface_hub"`.
            commit_description (`str` *optional*)
                The description of the generated commit
            create_pr (`boolean`, *optional*):
                Whether or not to create a Pull Request from `revision` with the changes.
                Defaults to `False`.
            parent_commit (`str`, *optional*):
                The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported.
                If specified and `create_pr` is `False`, the commit will fail if `revision` does not point to `parent_commit`.
                If specified and `create_pr` is `True`, the pull request will be created from `parent_commit`.
                Specifying `parent_commit` ensures the repo has not changed before committing the changes, and can be
                especially useful if the repo is updated / committed to concurrently.


        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://2.python-requests.org/en/master/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.
            - [`~utils.RevisionNotFoundError`]
              If the revision to download from cannot be found.
            - [`~utils.EntryNotFoundError`]
              If the file to download cannot be found.

        </Tip>

        """
        commit_message = (
            commit_message
            if commit_message is not None
            else f"Delete {path_in_repo} with huggingface_hub"
        )

        operations = [CommitOperationDelete(path_in_repo=path_in_repo)]

        return await self.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            operations=operations,
            revision=revision,
            commit_message=commit_message,
            commit_description=commit_description,
            create_pr=create_pr,
            parent_commit=parent_commit,
        )

    async def get_full_repo_name(
        self,
        model_id: str,
        *,
        organization: Optional[str] = None,
        token: Optional[str] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
    ) -> str:
        """
        Returns the repository name for a given model ID and optional
        organization.

        Args:
            model_id (`str`):
                The name of the model.
            organization (`str`, *optional*):
                If passed, the repository name will be in the organization
                namespace instead of the user namespace.
            token (`str`, *optional*):
                Deprecated in favor of `use_auth_token`. Will be removed in 0.12.0.
                The Hugging Face authentication token
            use_auth_token (`bool` or `str`, *optional*):
                Whether to use the `auth_token` provided from the
                `huggingface_hub` cli. If not logged in, a valid `auth_token`
                can be passed in as a string.

        Returns:
            `str`: The repository name in the user's namespace
            ({username}/{model_id}) if no organization is passed, and under the
            organization namespace ({organization}/{model_id}) otherwise.
        """
        if token is not None:
            warnings.warn(
                "`token` is deprecated and will be removed in 0.12.0. Use"
                " `use_auth_token` instead.",
                FutureWarning,
            )

        if token is None and use_auth_token:
            token, name = await self._validate_or_retrieve_token(use_auth_token)

        if organization is None:
            if "/" in model_id:
                username = model_id.split("/")[0]
            else:
                username = (await self.whoami(token=token))["name"]
            return f"{username}/{model_id}"
        else:
            return f"{organization}/{model_id}"

    @validate_hf_hub_args
    async def get_repo_discussions(
        self,
        repo_id: str,
        *,
        repo_type: Optional[str] = None,
        token: Optional[str] = None,
    ) -> AsyncIterator[Discussion]:
        """
        Fetches Discussions and Pull Requests for the given repo.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if fetching from a dataset or
                space, `None` or `"model"` if fetching from a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token).

        Returns:
            `AsyncIterator[Discussion]`: An iterator of [`Discussion`] objects.

        Example:
            Collecting all discussions of a repo in a list:

            ```python
            >>> from huggingface_hub import get_repo_discussions
            >>> discussions_list = list(get_repo_discussions(repo_id="bert-base-uncased"))
            ```

            Iterating over discussions of a repo:

            ```python
            >>> from huggingface_hub import get_repo_discussions
            >>> for discussion in get_repo_discussions(repo_id="bert-base-uncased"):
            ...     print(discussion.num, discussion.title)
            ```
        """
        if repo_type not in REPO_TYPES:
            raise ValueError(f"Invalid repo type, must be one of {REPO_TYPES}")
        if repo_type is None:
            repo_type = REPO_TYPE_MODEL
        repo_id = f"{repo_type}s/{repo_id}"
        if token is None:
            token = HfFolder.get_token()

        async def _fetch_discussion_page(page_index: int):
            path = f"{self.endpoint}/api/{repo_id}/discussions?p={page_index}"
            async with httpx.AsyncClient(follow_redirects=True) as client:
                resp = await client.get(
                    path,
                    headers={"Authorization": f"Bearer {token}"} if token else None,
                )
            hf_raise_for_status(resp)
            paginated_discussions = resp.json()
            total = paginated_discussions["count"]
            start = paginated_discussions["start"]
            discussions = paginated_discussions["discussions"]
            has_next = (start + len(discussions)) < total
            return discussions, has_next

        has_next, page_index = True, 0

        while has_next:
            discussions, has_next = await _fetch_discussion_page(page_index=page_index)
            for discussion in discussions:
                yield Discussion(
                    title=discussion["title"],
                    num=discussion["num"],
                    author=discussion.get("author", {}).get("name", "deleted"),
                    created_at=parse_datetime(discussion["createdAt"]),
                    status=discussion["status"],
                    repo_id=discussion["repo"]["name"],
                    repo_type=discussion["repo"]["type"],
                    is_pull_request=discussion["isPullRequest"],
                )
            page_index = page_index + 1

    @validate_hf_hub_args
    async def get_discussion_details(
        self,
        repo_id: str,
        discussion_num: int,
        *,
        repo_type: Optional[str] = None,
        token: Optional[str] = None,
    ) -> DiscussionWithDetails:
        """Fetches a Discussion's / Pull Request 's details from the Hub.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            discussion_num (`int`):
                The number of the Discussion or Pull Request . Must be a strictly positive integer.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)

        Returns: [`DiscussionWithDetails`]

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://2.python-requests.org/en/master/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        if not isinstance(discussion_num, int) or discussion_num <= 0:
            raise ValueError("Invalid discussion_num, must be a positive integer")
        if repo_type not in REPO_TYPES:
            raise ValueError(f"Invalid repo type, must be one of {REPO_TYPES}")
        if repo_type is None:
            repo_type = REPO_TYPE_MODEL
        repo_id = f"{repo_type}s/{repo_id}"
        if token is None:
            token = HfFolder.get_token()

        path = f"{self.endpoint}/api/{repo_id}/discussions/{discussion_num}"

        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(
                path,
                params={"diff": "1"},
                headers={"Authorization": f"Bearer {token}"} if token else None,
            )
        hf_raise_for_status(resp)

        discussion_details = resp.json()
        is_pull_request = discussion_details["isPullRequest"]

        target_branch = (
            discussion_details["changes"]["base"] if is_pull_request else None
        )
        conflicting_files = (
            discussion_details["filesWithConflicts"] if is_pull_request else None
        )
        merge_commit_oid = (
            discussion_details["changes"].get("mergeCommitId", None)
            if is_pull_request
            else None
        )

        return DiscussionWithDetails(
            title=discussion_details["title"],
            num=discussion_details["num"],
            author=discussion_details.get("author", {}).get("name", "deleted"),
            created_at=parse_datetime(discussion_details["createdAt"]),
            status=discussion_details["status"],
            repo_id=discussion_details["repo"]["name"],
            repo_type=discussion_details["repo"]["type"],
            is_pull_request=discussion_details["isPullRequest"],
            events=[deserialize_event(evt) for evt in discussion_details["events"]],
            conflicting_files=conflicting_files,
            target_branch=target_branch,
            merge_commit_oid=merge_commit_oid,
            diff=discussion_details.get("diff"),
        )

    @validate_hf_hub_args
    async def create_discussion(
        self,
        repo_id: str,
        title: str,
        *,
        token: str,
        description: Optional[str] = None,
        repo_type: Optional[str] = None,
        pull_request: bool = False,
    ) -> DiscussionWithDetails:
        """Creates a Discussion or Pull Request.

        Pull Requests created programmatically will be in `"draft"` status.

        Creating a Pull Request with changes can also be done at once with [`HfApi.create_commit`].

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            title (`str`):
                The title of the discussion. It can be up to 200 characters long,
                and must be at least 3 characters long. Leading and trailing whitespaces
                will be stripped.
            token (`str`):
                An authentication token (See https://huggingface.co/settings/token)
            description (`str`, *optional*):
                An optional description for the Pull Request.
                Defaults to `"Discussion opened with the huggingface_hub Python library"`
            pull_request (`bool`, *optional*):
                Whether to create a Pull Request or discussion. If `True`, creates a Pull Request.
                If `False`, creates a discussion. Defaults to `False`.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.

        Returns: [`DiscussionWithDetails`]

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://2.python-requests.org/en/master/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>"""
        if repo_type not in REPO_TYPES:
            raise ValueError(f"Invalid repo type, must be one of {REPO_TYPES}")
        if repo_type is None:
            repo_type = REPO_TYPE_MODEL
        full_repo_id = f"{repo_type}s/{repo_id}"
        token, _ = await self._validate_or_retrieve_token(token=token)
        if description is not None:
            description = description.strip()
        description = (
            description
            if description
            else (
                f"{'Pull Request' if pull_request else 'Discussion'} opened with the"
                " [huggingface_hub Python"
                " library](https://huggingface.co/docs/huggingface_hub)"
            )
        )

        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.post(
                f"{self.endpoint}/api/{full_repo_id}/discussions",
                json={
                    "title": title.strip(),
                    "description": description,
                    "pullRequest": pull_request,
                },
                headers={"Authorization": f"Bearer {token}"},
            )
        hf_raise_for_status(resp)
        num = resp.json()["num"]
        return await self.get_discussion_details(
            repo_id=repo_id,
            repo_type=repo_type,
            discussion_num=num,
            token=token,
        )

    @validate_hf_hub_args
    async def create_pull_request(
        self,
        repo_id: str,
        title: str,
        *,
        token: str,
        description: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> DiscussionWithDetails:
        """Creates a Pull Request . Pull Requests created programmatically will be in `"draft"` status.

        Creating a Pull Request with changes can also be done at once with [`HfApi.create_commit`];

        This is a wrapper around [`HfApi.create_discusssion`].

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            title (`str`):
                The title of the discussion. It can be up to 200 characters long,
                and must be at least 3 characters long. Leading and trailing whitespaces
                will be stripped.
            token (`str`):
                An authentication token (See https://huggingface.co/settings/token)
            description (`str`, *optional*):
                An optional description for the Pull Request.
                Defaults to `"Discussion opened with the huggingface_hub Python library"`
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.

        Returns: [`DiscussionWithDetails`]

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://2.python-requests.org/en/master/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>"""
        return await self.create_discussion(
            repo_id=repo_id,
            title=title,
            token=token,
            description=description,
            repo_type=repo_type,
            pull_request=True,
        )

    async def _post_discussion_changes(
        self,
        *,
        repo_id: str,
        discussion_num: int,
        resource: str,
        body: Optional[dict] = None,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> httpx.Response:
        """Internal utility to POST changes to a Discussion or Pull Request"""
        if not isinstance(discussion_num, int) or discussion_num <= 0:
            raise ValueError("Invalid discussion_num, must be a positive integer")
        if repo_type not in REPO_TYPES:
            raise ValueError(f"Invalid repo type, must be one of {REPO_TYPES}")
        if repo_type is None:
            repo_type = REPO_TYPE_MODEL
        repo_id = f"{repo_type}s/{repo_id}"
        token, _ = await self._validate_or_retrieve_token(token=token)

        path = f"{self.endpoint}/api/{repo_id}/discussions/{discussion_num}/{resource}"

        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.post(
                path,
                headers={"Authorization": f"Bearer {token}"},
                json=body,
            )
        hf_raise_for_status(resp)
        return resp

    @validate_hf_hub_args
    async def comment_discussion(
        self,
        repo_id: str,
        discussion_num: int,
        comment: str,
        *,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> DiscussionComment:
        """Creates a new comment on the given Discussion.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            discussion_num (`int`):
                The number of the Discussion or Pull Request . Must be a strictly positive integer.
            comment (`str`):
                The content of the comment to create. Comments support markdown formatting.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)

        Returns:
            [`DiscussionComment`]: the newly created comment


        Examples:
            ```python

            >>> comment = \"\"\"
            ... Hello @otheruser!
            ...
            ... # This is a title
            ...
            ... **This is bold**, *this is italic* and ~this is strikethrough~
            ... And [this](http://url) is a link
            ... \"\"\"

            >>> HfApi().comment_discussion(
            ...     repo_id="username/repo_name",
            ...     discussion_num=34
            ...     comment=comment
            ... )
            # DiscussionComment(id='deadbeef0000000', type='comment', ...)

            ```

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://2.python-requests.org/en/master/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        resp = await self._post_discussion_changes(
            repo_id=repo_id,
            repo_type=repo_type,
            discussion_num=discussion_num,
            token=token,
            resource="comment",
            body={"comment": comment},
        )
        return deserialize_event(resp.json()["newMessage"])

    @validate_hf_hub_args
    async def rename_discussion(
        self,
        repo_id: str,
        discussion_num: int,
        new_title: str,
        *,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> DiscussionTitleChange:
        """Renames a Discussion.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            discussion_num (`int`):
                The number of the Discussion or Pull Request . Must be a strictly positive integer.
            new_title (`str`):
                The new title for the discussion
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)

        Returns:
            [`DiscussionTitleChange`]: the title change event


        Examples:
            ```python
            >>> new_title = "New title, fixing a typo"
            >>> HfApi().rename_discussion(
            ...     repo_id="username/repo_name",
            ...     discussion_num=34
            ...     new_title=new_title
            ... )
            # DiscussionTitleChange(id='deadbeef0000000', type='title-change', ...)

            ```

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://2.python-requests.org/en/master/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        resp = await self._post_discussion_changes(
            repo_id=repo_id,
            repo_type=repo_type,
            discussion_num=discussion_num,
            token=token,
            resource="title",
            body={"title": new_title},
        )
        return deserialize_event(resp.json()["newTitle"])

    @validate_hf_hub_args
    async def change_discussion_status(
        self,
        repo_id: str,
        discussion_num: int,
        new_status: Literal["open", "closed"],
        *,
        token: Optional[str] = None,
        comment: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> DiscussionStatusChange:
        """Closes or re-opens a Discussion or Pull Request.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            discussion_num (`int`):
                The number of the Discussion or Pull Request . Must be a strictly positive integer.
            new_status (`str`):
                The new status for the discussion, either `"open"` or `"closed"`.
            comment (`str`, *optional*):
                An optional comment to post with the status change.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)

        Returns:
            [`DiscussionStatusChange`]: the status change event


        Examples:
            ```python
            >>> new_title = "New title, fixing a typo"
            >>> HfApi().rename_discussion(
            ...     repo_id="username/repo_name",
            ...     discussion_num=34
            ...     new_title=new_title
            ... )
            # DiscussionStatusChange(id='deadbeef0000000', type='status-change', ...)

            ```

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://2.python-requests.org/en/master/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        if new_status not in ["open", "closed"]:
            raise ValueError("Invalid status, valid statuses are: 'open' and 'closed'")
        body = {"status": new_status}
        if comment and comment.strip():
            body["comment"] = comment.strip()
        resp = await self._post_discussion_changes(
            repo_id=repo_id,
            repo_type=repo_type,
            discussion_num=discussion_num,
            token=token,
            resource="status",
            body=body,
        )
        return deserialize_event(resp.json()["newStatus"])

    @validate_hf_hub_args
    async def merge_pull_request(
        self,
        repo_id: str,
        discussion_num: int,
        *,
        token: str,
        comment: Optional[str] = None,
        repo_type: Optional[str] = None,
    ):
        """Merges a Pull Request.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            discussion_num (`int`):
                The number of the Discussion or Pull Request . Must be a strictly positive integer.
            comment (`str`, *optional*):
                An optional comment to post with the status change.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)

        Returns:
            [`DiscussionStatusChange`]: the status change event

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://2.python-requests.org/en/master/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        await self._post_discussion_changes(
            repo_id=repo_id,
            repo_type=repo_type,
            discussion_num=discussion_num,
            token=token,
            resource="merge",
            body={"comment": comment.strip()} if comment and comment.strip() else None,
        )

    @validate_hf_hub_args
    async def edit_discussion_comment(
        self,
        repo_id: str,
        discussion_num: int,
        comment_id: str,
        new_content: str,
        *,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
    ) -> DiscussionComment:
        """Edits a comment on a Discussion / Pull Request.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            discussion_num (`int`):
                The number of the Discussion or Pull Request . Must be a strictly positive integer.
            comment_id (`str`):
                The ID of the comment to edit.
            new_content (`str`):
                The new content of the comment. Comments support markdown formatting.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)

        Returns:
            [`DiscussionComment`]: the edited comment

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://2.python-requests.org/en/master/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        resp = await self._post_discussion_changes(
            repo_id=repo_id,
            repo_type=repo_type,
            discussion_num=discussion_num,
            token=token,
            resource=f"comment/{comment_id.lower()}/edit",
            body={"content": new_content},
        )
        return deserialize_event(resp.json()["updatedComment"])

    @validate_hf_hub_args
    async def hide_discussion_comment(
        self,
        repo_id: str,
        discussion_num: int,
        comment_id: str,
        *,
        token: str,
        repo_type: Optional[str] = None,
    ) -> DiscussionComment:
        """Hides a comment on a Discussion / Pull Request.

        <Tip warning={true}>
        Hidden comments' content cannot be retrieved anymore. Hiding a comment is irreversible.
        </Tip>

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            discussion_num (`int`):
                The number of the Discussion or Pull Request . Must be a strictly positive integer.
            comment_id (`str`):
                The ID of the comment to edit.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)

        Returns:
            [`DiscussionComment`]: the hidden comment

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://2.python-requests.org/en/master/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
        warnings.warn(
            "Hidden comments' content cannot be retrieved anymore. Hiding a comment is"
            " irreversible.",
            UserWarning,
        )
        resp = await self._post_discussion_changes(
            repo_id=repo_id,
            repo_type=repo_type,
            discussion_num=discussion_num,
            token=token,
            resource=f"comment/{comment_id.lower()}/hide",
        )
        return deserialize_event(resp.json()["updatedComment"])
