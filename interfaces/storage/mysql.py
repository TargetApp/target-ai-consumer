import mysql.connector

from . import Storage


class MySQLStorage(Storage):
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

    def store_mask(self, mask: bytes, report_id: int):
        cursor = self._get_cursor()
        cursor.execute(
            """INSERT INTO MASK
            (ID, DATA) VALUES (%s, %s)""",
            (report_id, mask),
        )
        self.connection.commit()
        cursor.close()

    def retrieve_weights(self, model_id: int) -> bytes:
        cursor = self._get_cursor()
        cursor.execute(
            "SELECT W.DATA FROM WEIGHTS W WHERE W.ID = %s",
            (model_id,),
        )
        weights = cursor.fetchone()[0]
        cursor.close()
        return weights

    def store_weights(self, weights: bytes, model_id: int):
        cursor = self._get_cursor()
        cursor.execute(
            """INSERT INTO WEIGHTS
            (ID, DATA) VALUES (%s, %s)""",
            (model_id, weights),
        )
        self.connection.commit()
        cursor.close()

    def store_image(self, image: bytes, image_id: int):
        cursor = self._get_cursor()
        cursor.execute(
            """INSERT INTO IMAGE
            (ID, DATA) VALUES (%s, %s)""",
            (image_id, image),
        )
        self.connection.commit()
        cursor.close()
