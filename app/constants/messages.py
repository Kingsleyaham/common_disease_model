from enum import Enum

class MESSAGE(str, Enum):
    CREATED = 'Resource created successfully'
    UPDATED = 'Resource updated successfully'
    DELETED = 'Resource deleted successfully'

    # Error messages
    INTERNAL_SERVER_ERROR = 'Internal server error'