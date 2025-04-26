import mysql.connector

from models import ProcessingModelType

from . import Queue, QueueElement


class MySQLQueue(Queue):
    def __init__(self, module_settings: dict):
        self.connection = mysql.connector.connect(
            host=module_settings["host"],
            port=module_settings["port"],
            user=module_settings["user"],
            password=module_settings["password"],
            database=module_settings["database"],
        )
        cursor = self.connection.cursor()
        cursor.execute("SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED")
        cursor.close()

    def _get_cursor(self, *args, **kwargs):
        """Get a cursor ensuring the connection is active.

        Returns
        -------
        MySQLCursorAbstract
            Cursor.
        """
        if not self.connection.is_connected():
            self.connection.reconnect()
        return self.connection.cursor(*args, **kwargs)

    def _enqueue_to_processing_queue(
        self,
        image_id: int,
        model_id: int,
        model_type: ProcessingModelType,
        report_id: int,
        image: bytes,
        generate_mask: bool | None = None,
    ):
        cursor = self._get_cursor()
        classification_report_id: int | None
        segmentation_report_id: int | None
        match model_type:
            case ProcessingModelType.CLASSIFICATION:
                classification_report_id = report_id
                segmentation_report_id = None
            case ProcessingModelType.SEGMENTATION:
                classification_report_id = None
                segmentation_report_id = report_id
            case _:
                raise ValueError(f"Invalid model type: {model_type}")
        cursor.execute(
            """INSERT INTO VALIDATION_QUEUE
            (IMAGE_ID, MODEL_ID, CLASSIFICATION_REPORT_ID, SEGMENTATION_REPORT_ID, IMAGE, GENERATE_MASK)
            VALUES (%s, %s, %s, %s, %s, %s)""",
            (
                image_id,
                model_id,
                classification_report_id,
                segmentation_report_id,
                image,
                generate_mask,
            ),
        )
        self.connection.commit()
        cursor.close()

    def dequeue_from_processing_queue(self) -> QueueElement | None:
        cursor = self._get_cursor(dictionary=True)
        cursor.execute("LOCK TABLE PROCESSING_QUEUE WRITE")
        cursor.execute(
            """SELECT ID,
            MODEL_ID,
            CLASSIFICATION_REPORT_ID,
            SEGMENTATION_REPORT_ID,
            IMAGE,
            GENERATE_MASK
            FROM PROCESSING_QUEUE
            ORDER BY ID ASC
            LIMIT 1"""
        )
        row = cursor.fetchone()
        if row is None:
            cursor.execute("UNLOCK TABLES")
            cursor.close()
            return None
        cursor.execute(
            "DELETE FROM PROCESSING_QUEUE WHERE ID = %s",
            (row["ID"],),
        )
        self.connection.commit()
        cursor.execute("UNLOCK TABLES")
        cursor.close()
        report_id, report_type = self.process_report_id(
            row["CLASSIFICATION_REPORT_ID"], row["SEGMENTATION_REPORT_ID"]
        )
        return QueueElement(
            row["ID"],
            row["MODEL_ID"],
            report_id,
            report_type,
            row["IMAGE"],
            row["GENERATE_MASK"],
        )

    def processing_queue_has_elements(self) -> bool:
        cursor = self._get_cursor()
        cursor.execute("SELECT COUNT(*) FROM PROCESSING_QUEUE")
        result = cursor.fetchone()
        cursor.close()
        return result[0] > 0

    def _enqueue_to_validation_queue(
        self,
        image_id: int,
        validation_model_id: int,
        model_id: int,
        model_type: ProcessingModelType,
        report_id: int,
        image: bytes,
        generate_mask: bool | None = None,
    ):
        cursor = self._get_cursor()
        classification_report_id: int | None
        segmentation_report_id: int | None
        match model_type:
            case ProcessingModelType.CLASSIFICATION:
                classification_report_id = report_id
                segmentation_report_id = None
            case ProcessingModelType.SEGMENTATION:
                classification_report_id = None
                segmentation_report_id = report_id
            case _:
                raise ValueError(f"Invalid model type: {model_type}")
        cursor.execute(
            """INSERT INTO VALIDATION_QUEUE
            (IMAGE_ID, VALIDATION_MODEL_ID, MODEL_ID, CLASSIFICATION_REPORT_ID, SEGMENTATION_REPORT_ID, IMAGE, GENERATE_MASK)
            VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (
                image_id,
                validation_model_id,
                model_id,
                classification_report_id,
                segmentation_report_id,
                image,
                generate_mask,
            ),
        )
        self.connection.commit()
        cursor.close()

    def dequeue_from_validation_queue(self) -> QueueElement | None:
        cursor = self._get_cursor(dictionary=True)
        cursor.execute("LOCK TABLES VALIDATION_QUEUE WRITE, BUFFER WRITE")
        cursor.execute(
            """SELECT ID,
            VALIDATION_MODEL_ID,
            CLASSIFICATION_REPORT_ID,
            SEGMENTATION_REPORT_ID,
            IMAGE
            FROM VALIDATION_QUEUE
            ORDER BY ID ASC
            LIMIT 1"""
        )
        row = cursor.fetchone()
        if row is None:
            cursor.execute("UNLOCK TABLES")
            cursor.close()
            return None
        cursor.execute(
            """INSERT INTO BUFFER
            (ID, IMAGE_ID, MODEL_ID, CLASSIFICATION_REPORT_ID, SEGMENTATION_REPORT_ID, IMAGE, GENERATE_MASK)
            SELECT ID, IMAGE_ID, MODEL_ID, CLASSIFICATION_REPORT_ID, SEGMENTATION_REPORT_ID, IMAGE, GENERATE_MASK
            FROM VALIDATION_QUEUE
            WHERE ID = %s""",
            (row["ID"],),
        )
        cursor.execute(
            "DELETE FROM VALIDATION_QUEUE WHERE ID = %s",
            (row["ID"],),
        )
        self.connection.commit()
        cursor.execute("UNLOCK TABLES")
        cursor.close()
        report_id, report_type = self.process_report_id(
            row["CLASSIFICATION_REPORT_ID"], row["SEGMENTATION_REPORT_ID"]
        )
        return QueueElement(
            row["ID"], row["VALIDATION_MODEL_ID"], report_id, report_type, row["IMAGE"]
        )

    def update_buffer(self, element_id: int, validation_result: bool):
        cursor = self._get_cursor()
        if validation_result:
            cursor.execute(
                """INSERT INTO PROCESSING_QUEUE
                (IMAGE_ID, MODEL_ID, CLASSIFICATION_REPORT_ID, SEGMENTATION_REPORT_ID, IMAGE, GENERATE_MASK)
                SELECT IMAGE_ID, MODEL_ID, CLASSIFICATION_REPORT_ID, SEGMENTATION_REPORT_ID, IMAGE, GENERATE_MASK
                FROM BUFFER
                WHERE ID = %s""",
                (element_id,),
            )
        cursor.execute(
            "DELETE FROM BUFFER WHERE ID = %s",
            (element_id,),
        )
        self.connection.commit()
        cursor.close()

    def validation_queue_has_elements(self) -> bool:
        cursor = self._get_cursor()
        cursor.execute("SELECT COUNT(*) FROM VALIDATION_QUEUE")
        result = cursor.fetchone()
        cursor.close()
        return result[0] > 0
