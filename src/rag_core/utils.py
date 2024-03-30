import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from fastapi import HTTPException, status
from label_studio_sdk import Client, Project
from loguru import logger

from app.config import settings
from ls_al.nlp_task_reformat import load_dict, min_conf_calc, task_type_schema


def get_client(token: str) -> Client:
    """
    Initialize LabelStudio Client
    Args:
        token (str): API key for accessing label studio
    Returns:
        label studio client
    """
    # Get labeled data
    try:
        ls_client = Client(url=settings.LABEL_STUDIO_HOST, api_key=token)
        # Istantiating the client doesnt check the connection. Below will check the connection
        # If no API KEY is passed it will use a session token and still pass
        # TODO: No sure if that poses an issue?
        ls_client.check_connection()
    except Exception as e:
        err_msg = (
            "It fails to connect to Label Studio with the given credentials."
            "Check the LSE URL or your credentials"
        )
        logger.error(err_msg, details=json.dumps({"message": str(e)}))
        raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED, detail=err_msg)
    else:
        return ls_client


def create_project(client: Client, task_type="ner", folder_path="../templates") -> Project:
    """Creating empty project
    Args:
        client (Client): Label Studio Client instance
        task_type (str): Name of the task
    Returns:
        label studio project
    """
    label_config_template = load_dict(f"{folder_path}/label_config_template.json")
    project_config_template = load_dict(f"{folder_path}/project_config_template.json")

    project_config_template["label_config"] = "\n".join(
        label_config_template[task_type]["project_label_config"]
    )
    proj = client.start_project(**project_config_template)
    return proj


def upload_data(proj: Project, data: List[Dict]):
    """Uploading data in the project
    Args:
        proj (Project): Label Studio project instance
        data: List of dictionary
    """
    proj.import_tasks(data)


def get_project_client(client: Client, project_id: int) -> Project:
    """
    Get LabelStudio project based on id
    Args:
        client (Client): Label studio Client
        project_id (int): Project id in labelstudio
    Returns:
        label studio project
    Raises:
        HTTPError: When the client/project is invalid
    """
    try:
        proj = client.get_project(id=project_id)
    except Exception as http_exception:
        logger.error(http_exception)
        raise Exception("HTTP Error, invalid client/project id")
    else:
        return proj


def get_all_data(client: Client, token: str, project_id: int = None) -> pd.DataFrame:
    """
    Get all records from labelstudio for the given project
    Args:
        client (Client): Label studio Client
        project_id (int): Project id in labelstudio
    Returns:
        data (pd.DataFrame) : DataFrame of annotated records
    """
    if project_id is None:
        raise ValueError("project_id can't be None")

    logger.info("Getting all tasks...")
    proj_url = f"{settings.LABEL_STUDIO_HOST}/api/projects"
    headers = {"Authorization": f"Token {token}"}
    proj_url_task = f"{proj_url}/{project_id}/tasks/?page_size=-1"
    try:
        tasks_resp = requests.get(proj_url_task, headers=headers).json()
    except requests.exceptions.RequestException as e:
        logger.error(e.response.text)
        raise SystemExit(e)
    else:
        tasks_df = pd.DataFrame(tasks_resp)
        logger.info(f"Data size of Project ID: {project_id} :: {tasks_df.shape}")
        return tasks_df


def get_label_config(project: Project, task_type: str = "ner") -> Tuple[str, str, str]:
    """
    Obtaining the label config from project. It has 2 keys that is required
    to be passed for proper formating the prediciton response back to LS.
    Args:
        project (Project) : project instance
    Returns:
        Tuple(str,str,str): For ner/cls/summary task, the value of "name" and "toName" in Labels/Choices/TextArea tag,  # noqa: E501
                            and the key/colname of the model input data.
    """
    # sample format of label config in XML
    # <View>\n
    #     <Text name="text" value="$text"/>\n
    #     <View>\n
    #         <Labels name="label" toName="text">\n
    #             <Label value="A"/>\n
    #             <Label value="B"/>\n
    #         </Labels>
    #     </View>\n
    # </View>
    # We need to obtain the fields name and toName

    # getting project label params
    project_config = project.get_params()
    xmlstring = project_config["label_config"]
    # getting the xml format
    tree = ET.ElementTree(ET.fromstring(xmlstring))

    name, toName, text_items = None, None, []
    task_tag_mapping = {"cls": "Choices", "ner": "Labels", "summary": "TextArea"}
    msg = f"Invalid labeling config. For {task_type} task, ensure there is one and only one {task_tag_mapping[task_type]} tag with `name` and `toName` attributes"  # noqa: E501
    for item in tree.iter():
        if item.tag == task_tag_mapping[task_type]:
            # assume there is only one Choices or Labels tag
            if name is None and toName is None:
                name = item.attrib["name"]
                toName = item.attrib["toName"]
            else:
                logger.error(msg)
                raise ValueError(f"Invalid label_config of project: {project.title}. {msg}")
        # UI dispaly content
        if item.tag == "Text":
            text_items.append(item)

    if name is None or toName is None:
        logger.error(msg)
        raise ValueError(f"Invalid label_config of project: {project.title}.{msg}")

    # detect the col name in the original dataset
    data_cols = []
    for text_tag in text_items:
        if text_tag.attrib["name"] in toName.split(","):
            col_name = text_tag.attrib["value"]
            if col_name[0] == "$":
                data_cols.append(col_name[1:])

    if len(data_cols) == 0:
        raise ValueError(
            f"Failed in detecting the key/colname of the model input data. Invalid label_config of project: {project.title}."  # noqa: E501
        )

    return (name, toName, ",".join(data_cols))


def format_endpoint_input(
    data: pd.DataFrame,
    task_type: str = "ner",
    model_input_text_attribute: str = "text",
) -> List:
    """
    Format the labelstudio data for endpoint prediction
    Args:
        data (pd.DataFrame) : DataFrame of annotated records
        task_type(str): task type,Defaults to "ner".
        model_input_text_attribute(str): which col/attribute is the model input data.  Defaults to "text".   # noqa: E501
    Returns:
        results (list) : List of data prepared for prediction with task id as their key.
    Raises:
        ValueError: If text key is not present in data
    """
    results = []
    if "," in model_input_text_attribute:
        model_input_text_attribute = model_input_text_attribute.split(",")

    # cls case:
    if task_type == "cls":
        text_cols = (
            [model_input_text_attribute]
            if isinstance(model_input_text_attribute, str)
            else model_input_text_attribute
        )
        part_data_df = data[["id", "data"]]
        part_data_df["model_input_data"] = part_data_df.apply(
            lambda r: {"text": " ".join([r.data[text_col] for text_col in text_cols])},
            axis=1,
        )
        results = part_data_df["model_input_data"].to_list()

    # summary case
    if task_type == "summary":
        pass

    # ner case
    if task_type == "ner":
        if "text" in data.iloc[0]["data"].keys():
            for uid, data in zip(data["id"].values, data["data"].values):
                tmp_dict = {}
                tmp_dict["id"] = str(uid)
                tmp_dict["text"] = data["text"]
                results.append(tmp_dict)
        else:
            raise ValueError("Text column not found. Only NLP usecases are supported as of now")

    logger.info("Formatted the data")
    logger.info(f"Sample data :: {results[:2]}")
    return results


def update_labelstudio_tasks(
    project: Project,
    prediction_response_sample: pd.DataFrame,
    task_data: pd.DataFrame,
    label_key: str = "ents",
    score_key: str = None,
    task_type: str = "ner",
    template_path: str = "../templates/label_config_template.json",
):
    """Converts the dataframe into labelstudio format
    Args:
        project (Project): class instance of the project selected from Label Studio
        prediction_response_sample (Dict): raw response from model endpoint
        task_data (pd.DataFrame): task data (records) obtained from project
        label_key (str): key containing predicition from endpoint response
        score_key (str): score key in the prediciton response
        task_type (str): kind of use case we are targeting.
    Raises:
        AssertionError: if the task_type are not correct

    """
    assert task_type in ["ner", "cls"]
    prediction_response_sample_json = prediction_response_sample.to_dict("records")

    if task_type == "cls":
        update_labelstudio_prediction_for_cls(
            project=project,
            task_ids=task_data["id"].tolist(),
            predictions=prediction_response_sample_json,
            label_key=label_key,
            score_key=score_key,
        )
        return

    # pls test when the value of to_name is not equal the original col name in the dataset
    # not all pii ner project has `text` col and the value of "toName" is `text` too
    if task_type == "ner":
        update_labelstudio_prediction_for_ner(
            project=project,
            task_ids=task_data["id"].tolist(),
            predictions=prediction_response_sample_json,
            label_key=label_key,
            template_path=template_path,
        )
        return


def update_labelstudio_prediction_for_ner(
    project: Project,
    task_ids: List[int],
    predictions: List[dict],
    label_key: str = "ents",
    score_key: Optional[str] = "score",
    template_path: str = "../templates/label_config_template.json",
) -> None:
    """Update the given task_ids' predictions in labelstuido project.

    Args:
        project (Project): Label Studio Project.
        task_ids(List[int]): The unique task ids in Label Studio project that need to be updated.
        predictions (List[dict]): the corespoindings predictions for all items in the `task_ids`.
        model_version:str  Defaults to None.
        label_key:str  Defaults to "predict_label". Label key in the server's prediction response.
        score_key:str  Defaults to "score". Score key in the in the server's prediction response.
    """
    assert len(predictions) == len(
        task_ids
    ), "Num of `predictions` doesn't match the num of `task_ids`"
    all_valid_task_ids = project.get_tasks_ids()

    from_label, to_label, _ = get_label_config(project)

    for prediction in predictions:
        task_id = prediction["id"]
        if int(task_id) in all_valid_task_ids:
            scores, pred_reformated = [], []
            for idx, each_field in enumerate(prediction[label_key]):
                pred, score = task_type_schema(template_path, idx, from_label, to_label, each_field)
                scores.append(score)
                pred_reformated.append(pred)

            min_conf_label_pair = min_conf_calc(scores)
            if min_conf_label_pair is not None:
                new_min_score = min([v for k, v in min_conf_label_pair.items()])

                project.create_prediction(
                    task_id=task_id,
                    result=pred_reformated,
                    score=new_min_score,
                    model_version=prediction.get("model_id", None),
                )
                logger.success(f"Prediction for Task ID: {task_id} Updated")
            else:
                logger.warning(f"No prediction for Task ID: {task_id}")
        else:
            logger.warning(
                f"{task_id} doesn't exist in porject {project.title} anymore. Skip this task id...."
            )


def update_labelstudio_prediction_for_cls(
    project: Project,
    task_ids: List[int],
    predictions: List[dict],
    label_key: str = "predict_label",
    score_key: str = "score",
) -> None:
    """Update the given task_ids' predictions in labelstuido project.

    Args:
        project (Project): Label Studio Project.
        task_ids(List[int]): The unique task ids in Label Studio project that need to be updated.
        predictions (List[dict]): the corespoindings predictions for all items in the `task_ids`.
        model_version:str  Defaults to None.
        label_key:str  Defaults to "predict_label". Label key in the server's prediction response.
        score_key:str  Defaults to "score". Score key in the in the server's prediction response.
    """
    assert len(predictions) == len(
        task_ids
    ), "Num of `predictions` doesn't match the num of `task_ids`"
    all_valid_task_ids = project.get_tasks_ids()

    for task_id, prediction in zip(task_ids, predictions):
        if task_id in all_valid_task_ids:
            result = prediction[label_key]
            score = prediction[score_key]
            project.create_prediction(
                task_id, result=result, score=score, model_version=prediction.get("model_id", None)
            )
            logger.success(f"Prediction for Task ID: {task_id} Updated")
        else:
            logger.warning(
                f"{task_id} doesn't exist in porject {project.title} anymore. Skip this task id...."
            )
