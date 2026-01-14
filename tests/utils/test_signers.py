import pandas as pd

from fatigue.utils import signers


def test_infer_signer():
    # 1. Create dummy input/output data (simulating your Fatigue data)
    # Inputs: Heart Rate (float), Sleep Hours (float)
    inputs = pd.DataFrame({"heart_rate": [70.5, 80.2, 65.0], "sleep_hours": [7.5, 6.0, 8.0]})

    # Outputs: Fatigue Score (float)
    outputs = pd.DataFrame({"fatigue_score": [0.2, 0.8, 0.1]})

    # 2. Initialize the Signer
    signer = signers.InferSigner()

    # 3. Generate the signature
    signature = signer.sign(inputs, outputs)

    # 4. Verify the signature structure (MLflow specific attributes)
    # Check inputs
    input_schema = signature.inputs
    assert "heart_rate" in input_schema.input_names()
    assert "sleep_hours" in input_schema.input_names()
    assert [str(t) for t in input_schema.input_types()] == [
        "DataType.double",
        "DataType.double",
    ]  # MLflow uses 'double' for float

    # Check outputs
    output_schema = signature.outputs
    assert output_schema.input_names() == ["fatigue_score"]

    print("Success! InferSigner generated a valid MLflow signature.")


if __name__ == "__main__":
    test_infer_signer()
