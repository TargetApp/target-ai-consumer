# target-ai-consumer

target-ai-consumer is a key component of Target-App, a web application designed for diagnosing biotic stress in coffee leaves through advanced image analysis. This project leverages artificial intelligence to assess the severity and type of biotic stress affecting coffee leaves. It supports two types of AI models: classification and segmentation. The classification model determines the type of biotic stress and its severity on a scale from 0 to 4. In contrast, the segmentation model produces an image with a black background, where healthy leaf regions are marked in green and areas affected by biotic stress are in red. This imagery is used to calculate the affected ratio (from 0 to 1) and severity (from 0 to 4), based on the stress ratio.

This project is based on [lara2018](https://github.com/esgario/lara2018), which provides foundational concepts and methodologies for analyzing coffee leaf images.

Additionally, the application supports a validation AI that processes an image and returns `1` if the image contains a coffee leaf and `0` otherwise. This is an optional validation step that ensures that only valid coffee leaf images are processed further by the system.

## Table of Contents

- [target-ai-consumer](#target-ai-consumer)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Requirements](#requirements)
    - [Database, Queue, and Storage](#database-queue-and-storage)
  - [Installation/Usage Guide](#installationusage-guide)
    - [Running Directly on Host Machine](#running-directly-on-host-machine)
    - [Running with Docker](#running-with-docker)
  - [Usage](#usage)
    - [main.py](#mainpy)
      - [Parameters](#parameters)
      - [Settings File](#settings-file)
      - [Running modes](#running-modes)
    - [insert\_model.py](#insert_modelpy)
      - [Parameters](#parameters-1)
    - [enqueue\_report.py](#enqueue_reportpy)
      - [Parameters](#parameters-2)
  - [Workflow](#workflow)
  - [Project Structure](#project-structure)

## Introduction

target-ai-consumer automates the processing of report requests queued for analysis. It interacts with databases to update reports based on the AI model's output and uploads segmentation masks to storage, if applicable. Developed with modularity in mind, it facilitates easy integration with various databases, queues, and storage systems. This project includes scripts for inserting AI models into the database/storage and processing requests in the database/queue, ensuring seamless operation and integration with Target-App.

## Requirements

- Python 3.12 or higher
- Poetry (for dependency management)
- Docker (for Docker-based installation)
- MySQL (For currently implemented database, queue, and storage interfaces)
- Trained AI models for classification and/or segmentation tasks (refer to [lara2018](https://github.com/esgario/lara2018))
- A trained validation AI model (optional)

### Database, Queue, and Storage
The project is designed to be agnostic to the database, queue, and storage systems, allowing for easy integration with various technologies. However, the implemented interfaces are tailored for MySQL (See [target-infra](https://github.com/TargetApp/target-infra) for MySQL setup). To use the project with other systems, you may need to implement the respective interfaces for your chosen database, queue, and storage solutions following the logic and structure provided in the existing codebase. Feel free to submit a pull request if you implement new interfaces. The project is designed to be modular and extensible, making it easy to adapt to different technologies as needed.

## Installation/Usage Guide

### Running Directly on Host Machine

1. Clone the repository
2. Install the required Python packages: `poetry install`
3. Set up the database, queue, and storage systems according to [target-infra](https://github.com/TargetApp/target-infra)
4. Configure the (`settings.toml`)[#settings-file] file to match your database, queue, and storage configuration
5. Insert the AI model into the database and storage using the `insert_model.py` script. See the [insert_model.py](#insert_modelpy) section for details on how to do this
6. Run `python main.py <settings> <running_mode>` to start the consumer
7. Enqueue processing requests using the `enqueue_report.py` script. See the [enqueue_report.py](#enqueue_reportpy) section for details on how to do this

### Running with Docker

1. Clone the repository
2. Build the Docker image: `docker build -t target-ai-consumer .`
3. Set up the database, queue, and storage systems according to [target-infra](https://github.com/TargetApp/target-infra)
4. Configure the `settings.toml` file to match your database, queue, and storage configuration.
5. Run the Docker container using the provided `docker-compose.yml` file at the root directory of the project. Make sure to have the `settings.toml` file and the `./data/` directory in the same directory where you run the command, as they will be mapped to the container. The `settings.toml` file is used to set the parameters for running, and the `./data/` directory might be used to insert models and enqueue test requests. Use the following command: `docker-compose up -d`
6. Insert the AI models into the database and storage using the `insert_model.py` script. See the [insert_model.py](#insert_modelpy) section for details on how to do this
7. Restart the container: `docker-compose restart`
8. Enqueue processing requests using the `enqueue_report.py` script. See the [enqueue_report.py](#enqueue_reportpy) section for details on how to do this

## Usage

### main.py
The main entry point for consuming queue elements and processing reports. This script is responsible for connecting to the database, queue, and storage systems, and it handles the logic for processing requests based on the specified running mode.

- **Directly on Host Machine:** `python main.py <settings> <running_mode>`
- **With Docker:** When the Docker container starts, it automatically runs `main.py settings.toml`. Use the key/flag `volumes` to ensure that the settings file is mounted at `/usr/src/target-ai-consumer/settings.toml` inside the container. The `running_mode` still needs to be specified. Use the key/flag `command` to set the running mode. See the [example docker-compose file](docker-compose.yml) for reference.

#### Parameters

- `settings`: Path to the settings file.
- `running_mode`: The running mode. It must be chosen between `validation`, `processing` and `both`.

#### Settings File

The settings file is crucial for configuring the application's connection to various interfaces such as the database, queue, and storage systems. See the [example settings.toml file](settings.toml) for reference.

#### Running modes

There are three running modes: `validation`, `processing`, and `both`.

- **validation:** The application consumes from the validation queue and enqueues valid processing requests on processing queues using a validation AI.
- **processing:** The application consumes from the processing queue and updates the respective row with the results in the report tables.
- **both:** The application executes both validation and processing functions. When there is nothing left to validate, it starts processing the processing requests. When there is nothing left to process, it returns to fetching from the validation queue.

### insert_model.py
Insert a model into the database and storage. This script is used to register AI models in the system, making them available for processing requests. It allows you to specify various parameters related to the model, including its type, subtype, module, class name, version, and weights. The weights must be provided as a `.pth` file, which contains the trained parameters necessary for the model to perform its tasks.

- **Directly on Host Machine:** `python3 insert_model.py <settings> <category> <type> <subtype> <module> <class_name> <version> <weights> [--enabled]`
- **With Docker:** `docker exec target-ai-consumer python3 insert_model.py <settings> <category> <type> <subtype> <module> <class_name> <version> <weights> [--enabled]`

#### Parameters

- `settings`: Path to settings file.
- `category`: Specifies the category of the model being loaded. It can be either `validation` or `processing`.
- `type`: Specifies the type of the model. It must be one of the predefined values in the `ModelType` enum,"classification" or "segmentation". This argument is used to categorize the model based on its functionality and architecture.
- `subtype`: Defines the subtype of the model, providing further classification within its main type. This is a free-form string that can be used to describe variations or specific configurations of the model type. It indicates the directory within `models/{type}` the model module is located.
- `module`: Indicates the module name where the model class is defined. This is used to dynamically import the model class for instantiation or other operations.
- `class_name`: The name of the class within the specified module that represents the model. This class name is used to instantiate the model object.
- `version`: A string representing the version of the model. This can be used to manage different iterations or updates of the same model.
- `weights`: The file path to the model's weights. This should be a path to a file that contains the trained weights necessary for the model to perform its task.
- `--enabled`: A flag that, when specified, marks the model as enabled. This can be used to control whether the model should be considered active and available for use without removing its configuration.

### enqueue_report.py
Enqueue a processing request. This script is used to add a processing request to the queue for the consumer to process.

- **Directly on Host Machine:** `python enqueue_report.py <settings> <user_id> <image> <processing_model_id> [--validation-model-id] <processing_model_type> [--generate-mask]`
- **With Docker:** `docker exec target-ai-consumer python enqueue_report.py <settings> <user_id> <image> <processing_model_id> [--validation-model-id] <processing_model_type> [--generate-mask]`

#### Parameters

- `settings`: Path to settings file.
- `user_id`: An integer representing the unique identifier of the user. This ID is used to associate the report and the processed image with a specific user in the database.
- `image`: The file path to the image that will be analyzed. This should be a path to an image file that the model will process to detect and classify biotic stress.
- `processing_model_id`: An integer representing the unique identifier of the model to be used for processing the image. This ID is used to select the appropriate AI model for image analysis.
- `validation_model_id`: An integer representing the unique identifier of the model to be used for validating the image. This ID is used to select the appropriate AI model for image validation.
- `processing_model_type`: The type of the processing model according to the expected output. It can be either `classification` or `segmentation`.
- `--generate-mask`: A flag that, when specified, instructs the system to generate a segmentation mask of the image. This mask highlights the areas affected by biotic stress, differentiating between healthy and stressed regions of the leaf. It's only available for segmentation models.

## Workflow

1. **Model Registration:** Register the model in the database and store its weights in storage to be loaded by the consumer at startup.
2. **User Registration:** Ensure there is a user registered in the database to be associated with the report.
3. **Image Information:** Add the image information (filename and user_id) to the database and upload the image to storage.
4. **Blank Report:** Add a blank report associated with the image to the database for it to be fulfilled by the consumer.
5. **Processing Request:** Include a processing request associated with the report in the queue.
6. **Input Validation:** If validation is enabled, the consumer will consume the validation request from the queue and validate the image using the validation model. If the image is valid, it will enqueue a processing request for the processing model.
7. **Processing Request:** The consumer will consume the processing request from the queue and process the image using the processing model. It will update the report in the database with the results and upload the segmentation mask to storage if applicable.

## Project Structure

The project structure is organized to facilitate easy navigation and understanding of the components involved in the target-ai-consumer application. Below is a brief overview of the main directories and files:

- `core/`: Contains the core logic for classification, segmentation and validation models.
- `interfaces/`: Defines interfaces for database, queue, and storage systems.
- `models/`: Includes definitions for model types and utilities.
- `main.py`: The main entry point for consuming queue elements and processing reports.
- `insert_model.py`: A script for inserting AI models into the database and storage.
- `enqueue_report.py`: A script for enqueuing processing requests.
- `settings.toml`: Configuration file for database, queue, and storage settings.
- `docker-compose.yml`: Docker Compose file for running the application in a containerized environment.

For more detailed information on each component, refer to the respective source files and documentation within the project.