from pytest_mock import MockerFixture

from src.api.query import get_similar_responses
from src.models.query import RAGRequest, RAGResponseItem


def test_query_endpoint(mocker: MockerFixture):
    mock_response = [
        RAGResponseItem(question="test question 1", wiki_excerpt="test answer 1")
    ]

    mocker.patch(
        "src.api.query.retriever.get_similar_responses", return_value=mock_response
    )

    mock_request = RAGRequest(question="What's this test for?", num_responses=1)
    result = get_similar_responses(request=mock_request)

    assert len(result.answers) == 1
    assert result.answers[0].question == "test question 1"


def test_query_endpoint_more_responses(mocker: MockerFixture):
    mock_response = [
        RAGResponseItem(question="test question 1", wiki_excerpt="test answer 1"),
        RAGResponseItem(question="test question 2", wiki_excerpt="test answer 2"),
        RAGResponseItem(question="test question 3", wiki_excerpt="test answer 3"),
        RAGResponseItem(question="test question 4", wiki_excerpt="test answer 4"),
        RAGResponseItem(question="test question 5", wiki_excerpt="test answer 5"),
    ]

    mocker.patch(
        "src.api.query.retriever.get_similar_responses", return_value=mock_response
    )

    mock_request = RAGRequest(question="What's this test for?", num_responses=3)
    result = get_similar_responses(request=mock_request)

    assert len(result.answers) == 5
    assert result.answers[0].question == "test question 1"
    assert result.answers[2].question == "test question 3"
    assert result.answers[3].question == "test question 4"
    assert result.answers[4].question == "test question 5"
    assert result.answers[0].wiki_excerpt == "test answer 1"
    assert result.answers[2].wiki_excerpt == "test answer 3"
    assert result.answers[3].wiki_excerpt == "test answer 4"
    assert result.answers[4].wiki_excerpt == "test answer 5"
