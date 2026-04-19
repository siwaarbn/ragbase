import os
from dataclasses import dataclass
from dotenv import load_dotenv
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

load_dotenv()


@dataclass
class EvalScores:
    faithfulness: float
    answer_relevancy: float


def score_answer(question: str, answer: str, contexts: list[str]) -> EvalScores:
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )
    dataset = EvaluationDataset(samples=[sample])
    llm = _get_judge_llm()
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy()],
        llm=llm,
        show_progress=False,
        raise_exceptions=False,
    )
    scores = result.to_pandas()
    return EvalScores(
        faithfulness=round(float(scores["faithfulness"].iloc[0]), 3),
        answer_relevancy=round(float(scores["answer_relevancy"].iloc[0]), 3),
    )


def _get_judge_llm():
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
    )
