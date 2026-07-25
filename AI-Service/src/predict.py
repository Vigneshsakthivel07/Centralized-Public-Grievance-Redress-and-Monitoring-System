import json
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


MODEL_PATH = "models/complaint_classifier"

CONFIDENCE_THRESHOLD = 0.70


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(device)
model.eval()


# ============================================================
# LABEL MAPPING
# ============================================================

with open(
    f"{MODEL_PATH}/label_mapping.json",
    "r",
    encoding="utf-8"
) as file:

    mapping = json.load(file)


id2label = {
    int(k): v
    for k, v in mapping["id2label"].items()
}


# ============================================================
# PREDICTION
# ============================================================

def predict_department(
    complaint: str
):

    if not isinstance(
        complaint,
        str
    ):

        raise TypeError(
            "Complaint must be a string."
        )


    complaint = complaint.strip()


    if not complaint:

        raise ValueError(
            "Complaint cannot be empty."
        )


    # --------------------------------------------------------
    # TOKENIZATION
    # --------------------------------------------------------

    encoding = tokenizer(
        complaint,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )


    encoding = {
        key: value.to(device)
        for key, value in encoding.items()
    }


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            **encoding
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )


    # --------------------------------------------------------
    # TOP 3
    # --------------------------------------------------------

    values, indices = torch.topk(
        probabilities,
        k=3,
        dim=-1
    )


    values = values[0].cpu().tolist()
    indices = indices[0].cpu().tolist()


    predicted_id = indices[0]

    predicted_department = (
        id2label[predicted_id]
    )

    confidence = values[0]


    top_predictions = []

    for index, probability in zip(
        indices,
        values
    ):

        top_predictions.append({
            "department": id2label[index],
            "confidence": round(
                probability,
                4
            )
        })


    # --------------------------------------------------------
    # MANUAL REVIEW
    # --------------------------------------------------------

    needs_manual_review = (
        confidence < CONFIDENCE_THRESHOLD
    )


    return {

        "complaint": complaint,

        "department": predicted_department,

        "confidence": round(
            confidence,
            4
        ),

        "needs_manual_review":
            needs_manual_review,

        "top_predictions":
            top_predictions
    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    complaints = [

        "There is no drinking water in our street.",

        "Huge potholes are causing accidents near our school.",

        "Street lights have not been working for two weeks.",

        "There are frequent power cuts in our area.",

        "Garbage has not been collected for several days.",

        "A doctor is not available at the government hospital.",

        "The traffic signal is not working.",

        "The government bus service is always delayed.",

        "An officer demanded a bribe from me.",

        "The school building is unsafe and damaged.",

    ]


    for complaint in complaints:

        result = predict_department(
            complaint
        )

        print("\n" + "=" * 70)

        print(
            "Complaint:",
            result["complaint"]
        )

        print(
            "Department:",
            result["department"]
        )

        print(
            "Confidence:",
            result["confidence"]
        )

        print(
            "Manual Review:",
            result["needs_manual_review"]
        )

        print(
            "Top predictions:"
        )

        for prediction in (
            result["top_predictions"]
        ):

            print(
                f"  {prediction['department']}: "
                f"{prediction['confidence']:.2%}"
            )