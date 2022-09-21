import os
from os.path import expanduser
from typing import Dict, List, Optional

from .utils._typing import TypedDict


class BlobLfsInfo(TypedDict, total=False):
    size: int
    sha256: str


class RepoFile:
    """
    Data structure that represents a public file inside a repo, accessible from
    huggingface.co

    Args:
        rfilename (str):
            file name, relative to the repo root. This is the only attribute
            that's guaranteed to be here, but under certain conditions there can
            certain other stuff.
        size (`int`, *optional*):
            The file's size, in bytes. This attribute is present when `files_metadata` argument
            of [`repo_info`] is set to `True`. It's `None` otherwise.
        blob_id (`str`, *optional*):
            The file's git OID. This attribute is present when `files_metadata` argument
            of [`repo_info`] is set to `True`. It's `None` otherwise.
        lfs (`BlobLfsInfo`, *optional*):
            The file's LFS metadata. This attribute is present when`files_metadata` argument
            of [`repo_info`] is set to `True` and the file is stored with Git LFS. It's `None` otherwise.
    """

    def __init__(
        self,
        rfilename: str,
        size: Optional[int] = None,
        blobId: Optional[str] = None,
        lfs: Optional[BlobLfsInfo] = None,
        **kwargs,
    ):
        self.rfilename = rfilename  # filename relative to the repo root

        # Optional file metadata
        self.size = size
        self.blob_id = blobId
        self.lfs = lfs

        # Hack to ensure backward compatibility with future versions of the API.
        # See discussion in https://github.com/huggingface/huggingface_hub/pull/951#discussion_r926460408
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        items = (f"{k}='{v}'" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({', '.join(items)})"


class ModelInfo:
    """
    Info about a model accessible from huggingface.co

    Attributes:
        modelId (`str`, *optional*):
            ID of model repository.
        sha (`str`, *optional*):
            repo sha at this particular revision
        lastModified (`str`, *optional*):
            date of last commit to repo
        tags (`List[str]`, *optional*):
            List of tags.
        pipeline_tag (`str`, *optional*):
            Pipeline tag to identify the correct widget.
        siblings (`List[RepoFile]`, *optional*):
            list of ([`huggingface_hub.hf_api.RepoFile`]) objects that constitute the model.
        private (`bool`, *optional*, defaults to `False`):
            is the repo private
        author (`str`, *optional*):
            repo author
        config (`Dict`, *optional*):
            Model configuration information
        securityStatus (`Dict`, *optional*):
            Security status of the model.
            Example: `{"containsInfected": False}`
        kwargs (`Dict`, *optional*):
            Kwargs that will be become attributes of the class.
    """

    def __init__(
        self,
        *,
        modelId: Optional[str] = None,
        sha: Optional[str] = None,
        lastModified: Optional[str] = None,
        tags: Optional[List[str]] = None,
        pipeline_tag: Optional[str] = None,
        siblings: Optional[List[Dict]] = None,
        private: bool = False,
        author: Optional[str] = None,
        config: Optional[Dict] = None,
        securityStatus: Optional[Dict] = None,
        **kwargs,
    ):
        self.modelId = modelId
        self.sha = sha
        self.lastModified = lastModified
        self.tags = tags
        self.pipeline_tag = pipeline_tag
        self.siblings = (
            [RepoFile(**x) for x in siblings] if siblings is not None else None
        )
        self.private = private
        self.author = author
        self.config = config
        self.securityStatus = securityStatus
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        s = f"{self.__class__.__name__}:" + " {"
        for key, val in self.__dict__.items():
            s += f"\n\t{key}: {val}"
        return s + "\n}"

    def __str__(self):
        r = f"Model Name: {self.modelId}, Tags: {self.tags}"
        if self.pipeline_tag:
            r += f", Task: {self.pipeline_tag}"
        return r


class DatasetInfo:
    """
    Info about a dataset accessible from huggingface.co

    Attributes:
        id (`str`, *optional*):
            ID of dataset repository.
        sha (`str`, *optional*):
            repo sha at this particular revision
        lastModified (`str`, *optional*):
            date of last commit to repo
        tags (`Listr[str]`, *optional*):
            List of tags.
        siblings (`List[RepoFile]`, *optional*):
            list of [`huggingface_hub.hf_api.RepoFile`] objects that constitute the dataset.
        private (`bool`, *optional*, defaults to `False`):
            is the repo private
        author (`str`, *optional*):
            repo author
        description (`str`, *optional*):
            Description of the dataset
        citation (`str`, *optional*):
            Dataset citation
        cardData (`Dict`, *optional*):
            Metadata of the model card as a dictionary.
        kwargs (`Dict`, *optional*):
            Kwargs that will be become attributes of the class.
    """

    def __init__(
        self,
        *,
        id: Optional[str] = None,
        sha: Optional[str] = None,
        lastModified: Optional[str] = None,
        tags: Optional[List[str]] = None,
        siblings: Optional[List[Dict]] = None,
        private: bool = False,
        author: Optional[str] = None,
        description: Optional[str] = None,
        citation: Optional[str] = None,
        cardData: Optional[dict] = None,
        **kwargs,
    ):
        self.id = id
        self.sha = sha
        self.lastModified = lastModified
        self.tags = tags
        self.private = private
        self.author = author
        self.description = description
        self.citation = citation
        self.cardData = cardData
        self.siblings = (
            [RepoFile(**x) for x in siblings] if siblings is not None else None
        )
        # Legacy stuff, "key" is always returned with an empty string
        # because of old versions of the datasets lib that need this field
        kwargs.pop("key", None)
        # Store all the other fields returned by the API
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        s = f"{self.__class__.__name__}:" + " {"
        for key, val in self.__dict__.items():
            s += f"\n\t{key}: {val}"
        return s + "\n}"

    def __str__(self):
        r = f"Dataset Name: {self.id}, Tags: {self.tags}"
        return r


class SpaceInfo:
    """
    Info about a Space accessible from huggingface.co

    This is a "dataclass" like container that just sets on itself any attribute
    passed by the server.

    Attributes:
        id (`str`, *optional*):
            id of space
        sha (`str`, *optional*):
            repo sha at this particular revision
        lastModified (`str`, *optional*):
            date of last commit to repo
        siblings (`List[RepoFile]`, *optional*):
            list of [`huggingface_hub.hf_api.RepoFIle`] objects that constitute the Space
        private (`bool`, *optional*, defaults to `False`):
            is the repo private
        author (`str`, *optional*):
            repo author
        kwargs (`Dict`, *optional*):
            Kwargs that will be become attributes of the class.
    """

    def __init__(
        self,
        *,
        id: Optional[str] = None,
        sha: Optional[str] = None,
        lastModified: Optional[str] = None,
        siblings: Optional[List[Dict]] = None,
        private: bool = False,
        author: Optional[str] = None,
        **kwargs,
    ):
        self.id = id
        self.sha = sha
        self.lastModified = lastModified
        self.siblings = (
            [RepoFile(**x) for x in siblings] if siblings is not None else None
        )
        self.private = private
        self.author = author
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        s = f"{self.__class__.__name__}:" + " {"
        for key, val in self.__dict__.items():
            s += f"\n\t{key}: {val}"
        return s + "\n}"


class MetricInfo:
    """
    Info about a public metric accessible from huggingface.co
    """

    def __init__(
        self,
        *,
        id: Optional[str] = None,  # id of metric
        description: Optional[str] = None,
        citation: Optional[str] = None,
        **kwargs,
    ):
        self.id = id
        self.description = description
        self.citation = citation
        # Legacy stuff, "key" is always returned with an empty string
        # because of old versions of the datasets lib that need this field
        kwargs.pop("key", None)
        # Store all the other fields returned by the API
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        s = f"{self.__class__.__name__}:" + " {"
        for key, val in self.__dict__.items():
            s += f"\n\t{key}: {val}"
        return s + "\n}"

    def __str__(self):
        r = f"Metric Name: {self.id}"
        return r


class HfFolder:
    path_token = expanduser("~/.huggingface/token")

    @classmethod
    def save_token(cls, token):
        """
        Save token, creating folder as needed.

        Args:
            token (`str`):
                The token to save to the [`HfFolder`]
        """
        os.makedirs(os.path.dirname(cls.path_token), exist_ok=True)
        with open(cls.path_token, "w+") as f:
            f.write(token)

    @classmethod
    def get_token(cls) -> Optional[str]:
        """
        Get token or None if not existent.

        Note that a token can be also provided using the
        `HUGGING_FACE_HUB_TOKEN` environment variable.

        Returns:
            `str` or `None`: The token, `None` if it doesn't exist.

        """
        token: Optional[str] = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token is None:
            try:
                with open(cls.path_token, "r") as f:
                    token = f.read()
            except FileNotFoundError:
                pass
        return token

    @classmethod
    def delete_token(cls):
        """
        Deletes the token from storage. Does not fail if token does not exist.
        """
        try:
            os.remove(cls.path_token)
        except FileNotFoundError:
            pass
