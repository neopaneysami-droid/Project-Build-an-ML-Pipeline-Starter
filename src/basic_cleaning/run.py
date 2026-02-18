#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact.
"""

import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# DO NOT MODIFY
def go(
    input_artifact: str,
    output_artifact: str,
    output_type: str,
    output_description: str,
    min_price: int,
    max_price: int
) -> None:
    """
    Run the basic cleaning step.

    Parameters
    ----------
    input_artifact : str
        Name of the raw data artifact to download from W&B.
    output_artifact : str
        Name of the cleaned data artifact to upload.
    output_type : str
        Type of the output artifact (e.g., 'clean_data').
    output_description : str
        Description of the output artifact.
    min_price : int
        Minimum allowed price for listings.
    max_price : int
        Maximum allowed price for listings.

    Returns
    -------
    None
    """

    run = wandb.init(job_type="basic_cleaning")
    run.config.update({
        "input_artifact": input_artifact,
        "output_artifact": output_artifact,
        "output_type": output_type,
        "output_description": output_description,
        "min_price": min_price,
        "max_price": max_price
    })

    # Download input artifact. This will also log that this script is using this
    run = wandb.init(project="nyc_airbnb", group="cleaning", save_code=True)
    artifact_local_path = run.use_artifact(input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    # Drop outliers
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Step 6: TODO
    # Only implement this step when reaching Step 6: Pipeline Release and Updates
    # in the project.
    # Add longitude and latitude filter to allow test_proper_boundaries to pass
    # ENTER CODE HERE

    # Save the cleaned data
    df.to_csv('clean_sample.csv', index=False)

    # log the new data.
    artifact = wandb.Artifact(
        output_artifact,
        type=output_type,
        description=output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)


# TODO: In the code below, fill in the data type for each argument. The data type should be str, float or int.
# TODO: In the code below, fill in a description for each argument. The description should be a string.
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the input raw data artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the output cleaned data artifact",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=int,
        help="Minimum allowed price for listings",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=int,
        help="Maximum allowed price for listings",
        required=True
    )

    args = parser.parse_args()

    go(
        args.input_artifact,
        args.output_artifact,
        args.output_type,
        args.output_description,
        args.min_price,
        args.max_price
    )
