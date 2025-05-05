from langchain_core.tools import tool
from datetime import datetime
import pytz
from typing import Optional, Annotated
from langchain_core.tools import render_text_description
from pydantic import BaseModel, Field

class GetCurrentTimeInput(BaseModel):
    timezone: Optional[str] = Field(default=None, description="A string representing the timezone (e.g., 'UTC', 'America/Los_Angeles').")

class ConvertTimeInput(BaseModel):
    source_time: str = Field(
        ...,
        description="The source time in the format '%Y-%m-%d %H:%M:%S %z', e.g., '2025-04-28 07:23:45 +0800'."
    )
    target_timezone: str = Field(
        ...,
        description="The IANA timezone name to convert the time into, e.g., 'America/New_York'."
    )

@tool(args_schema=GetCurrentTimeInput)
def get_current_time(
        timezone: Optional[str] = None
    ) -> str:
    """Gets the current date and time formatted as a string.

    Retrieves the current date and time for the specified timezone.
    If no timezone is provided, the system's local time is used.
    
    Returns:
        str: The current date and time formatted as 'YYYY-MM-DD HH:MM:SS <offset>'.

    Examples:
        >>> get_current_time('UTC')
        '2025-04-28 14:23:45 +0000'
        >>> get_current_time()
        '2025-04-28 07:23:45 -0700'
    """
    if timezone:
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)
    else:
        current_time = datetime.now().astimezone()

    return current_time.strftime("%Y-%m-%d %H:%M:%S %z")

@tool(args_schema=ConvertTimeInput)
def convert_time(source_time: str, target_timezone: str) -> str:
    """Converts a given time string to a specified target timezone.

    Returns:
        str: The converted time as a string in the format '%Y-%m-%d %H:%M:%S %z'.

    Raises:
        ValueError: If the source_time format is invalid or the target timezone is invalid.

    Example:
        >>> convert_time('2025-04-28 07:23:45 +0800', 'America/New_York')
        '2025-04-27 19:23:45 -0400'
    """
    try:
        dt = datetime.strptime(source_time, "%Y-%m-%d %H:%M:%S %z")
        target_tz = pytz.timezone(target_timezone)
        dt_in_target = dt.astimezone(target_tz)
        return dt_in_target.strftime("%Y-%m-%d %H:%M:%S %z")
    except Exception as e:
        raise ValueError(f"Failed to convert time: {e}")

if __name__ == '__main__':
    rendered_tools = render_text_description([get_current_time, convert_time])
    print(rendered_tools)
    print(get_current_time.args_schema.model_json_schema())
    print('-'*8)
    print(convert_time.args_schema.model_json_schema())