from typing import cast

import mysql.connector

from models import Model, ModelCategory
from models.processing import ProcessingModelType
from models.validation import ValidationModelType

from . import Database


class MySQLDatabase(Database):
    """Implementation of the database interface for MySQL.

    Parameters
    ----------
    DatabaseInterface : _abc.ABCMeta
        Abstract base class.
    """

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
        self._model_category_dict = self._get_dict_from_database(
            "MODEL_CATEGORY", "NAME", "ID"
        )
        self._model_type_dict = self._get_dict_from_database("MODEL_TYPE", "NAME", "ID")
        self._enabled_models = self._get_enabled_models()

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

    def update_classification_report(
        self, report_id: int, disease_id: int, severity_id: int
    ):
        cursor = self._get_cursor()
        cursor.execute(
            """UPDATE CLASSIFICATION_REPORT
            SET DISEASE_ID = %s,
            SEVERITY_ID = %s,
            PROCESSED_AT = NOW()
            WHERE ID = %s""",
            (disease_id, severity_id, report_id),
        )
        self.connection.commit()
        cursor.close()

    def update_segmentation_report(
        self, report_id: int, stress_ratio: float, severity_id: int
    ):
        cursor = self._get_cursor()
        cursor.execute(
            """UPDATE SEGMENTATION_REPORT
            SET STRESS_RATIO = %s,
            SEVERITY_ID = %s,
            PROCESSED_AT = NOW()
            WHERE ID = %s""",
            (stress_ratio, severity_id, report_id),
        )
        self.connection.commit()
        cursor.close()

    def _get_enabled_models(self) -> dict[int, Model]:
        """Get enabled models from the database and return them as a dictionary with the model ID as the key.

        Returns
        -------
        dict[int, Model]
            Enabled models.
        """
        cursor = self._get_cursor(dictionary=True)
        cursor.execute(
            """SELECT M.ID,
            MC.NAME AS CATEGORY,
            MT.NAME AS TYPE,
            M.SUBTYPE,
            M.MODULE,
            M.CLASS
            FROM MODEL M
            JOIN MODEL_CATEGORY MC ON M.MODEL_CATEGORY_ID = MC.ID
            JOIN MODEL_TYPE MT ON M.MODEL_TYPE_ID = MT.ID
            WHERE M.ENABLED = TRUE"""
        )
        models = dict()
        for row in cursor.fetchall():
            model_type_enum = ModelCategory[row["CATEGORY"]].value
            models[row["ID"]] = Model(
                category=ModelCategory[row["CATEGORY"]],
                type=cast(
                    ValidationModelType | ProcessingModelType,
                    model_type_enum[row["TYPE"]],
                ),
                subtype=row["SUBTYPE"],
                module=row["MODULE"],
                class_name=row["CLASS"],
            )
        cursor.close()
        return models

    def _get_dict_from_database(
        self, table_name: str, key_column: str, value_column: str
    ) -> dict:
        """Get a dictionary from the database.

        Parameters
        ----------
        table_name : str
            Name of the table.
        key_column : str
            Name of the key column.
        value_column : str
            Name of the value column.

        Returns
        -------
        dict
            Key-value pairs.
        """
        cursor = self._get_cursor()
        cursor.execute(f"SELECT {key_column}, {value_column} FROM {table_name}")
        result_dict = dict()
        for row in cursor.fetchall():
            result_dict[row[0]] = row[1]
        cursor.close()
        return result_dict

    @property
    def model_category_dict(self) -> dict[str, int]:
        return self._model_category_dict

    @property
    def model_type_dict(self) -> dict[str, int]:
        return self._model_type_dict

    @property
    def enabled_models(self) -> dict[int, Model]:
        return self._enabled_models

    def _insert_model(
        self,
        model_category: ModelCategory,
        model_type: ProcessingModelType | ValidationModelType,
        subtype: str,
        module: str,
        class_name: str,
        version: str,
        enabled: bool,
    ) -> int:
        cursor = self._get_cursor()
        model_category_id = self.model_category_dict[model_category.name]
        type_id = self.model_type_dict[model_type.name]
        cursor.execute(
            """INSERT INTO MODEL
            (MODEL_CATEGORY_ID, MODEL_TYPE_ID, SUBTYPE, MODULE, CLASS, VERSION, ENABLED)
            VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (model_category_id, type_id, subtype, module, class_name, version, enabled),
        )
        self.connection.commit()
        model_id: int = cursor.lastrowid
        cursor.close()
        return model_id

    def insert_classification_report(
        self, user_id: int, image_id: int, model_id: int
    ) -> int:
        cursor = self._get_cursor()
        cursor.execute(
            """INSERT INTO CLASSIFICATION_REPORT
            (USER_ID, IMAGE_ID, MODEL_ID)
            VALUES (%s, %s, %s)""",
            (user_id, image_id, model_id),
        )
        self.connection.commit()
        report_id: int = cursor.lastrowid
        cursor.close()
        return report_id

    def insert_segmentation_report(
        self, user_id: int, image_id: int, model_id: int, has_mask: bool = False
    ) -> int:
        cursor = self.connection.cursor()
        cursor.execute(
            """INSERT INTO SEGMENTATION_REPORT
            (USER_ID, IMAGE_ID, MODEL_ID, HAS_MASK)
            VALUES (%s, %s, %s, %s)""",
            (user_id, image_id, model_id, has_mask),
        )
        self.connection.commit()
        report_id: int = cast(int, cursor.lastrowid)
        cursor.close()
        return report_id

    def insert_image(self, user_id: int, filename: str) -> int:
        cursor = self._get_cursor()
        cursor.execute(
            """INSERT INTO IMAGE
            (USER_ID, FILENAME)
            VALUES (%s, %s)""",
            (user_id, filename),
        )
        self.connection.commit()
        image_id: int = cursor.lastrowid
        cursor.close()
        return image_id

    def update_report_validity(
        self, report_id: int, report_type: ProcessingModelType, valid: bool
    ):
        cursor = self._get_cursor()
        match report_type:
            case ProcessingModelType.CLASSIFICATION:
                report_table = "CLASSIFICATION_REPORT"
            case ProcessingModelType.SEGMENTATION:
                report_table = "SEGMENTATION_REPORT"
            case _:
                raise ValueError(f"Invalid report type: {report_type}")
        cursor.execute(
            f"UPDATE {report_table} SET VALID = %s WHERE ID = %s", (valid, report_id)
        )
        self.connection.commit()
        cursor.close()
