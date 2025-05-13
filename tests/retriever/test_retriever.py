from unittest.mock import MagicMock

from pytest_mock import MockerFixture

from src.models.query import RAGResponseItem
from src.retriever.retriever import get_similar_responses


def test_get_similar_responses(mocker: MockerFixture):
    mocker.patch("src.retriever.retriever._convert_question", return_value=MagicMock())
    mocker.patch("src.retriever.retriever._compute_similarity", return_value=MagicMock())
    mock_results = [RAGResponseItem(question="test question", wiki_excerpt="test excerpt")]
    mocker.patch("src.retriever.retriever._format_results", return_value=mock_results)



    result = get_similar_responses("test question", 3)
    assert len(result) == 1
    assert result[0].question == "test question"

