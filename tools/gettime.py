"""
Example OpenWebUI tool for getting current date and time.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel


class Tools:
    """Time and date utility tools."""

    class Valves(BaseModel):
        """Tool-level configuration (none needed for this tool)."""

        pass

    class UserValves(BaseModel):
        """User-specific configuration (none needed for this tool)."""

        pass

    def __init__(self):
        """Initialize tool with default valves."""
        self.valves = self.Valves()
        self.user_valves = self.UserValves()

    def get_current_date(self) -> str:
        """
        Get the current date.

        :return: The current date as a string.
        """
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        return f"Today's date is {current_date}"

    def get_current_time(self) -> str:
        """
        Get the current time.

        :return: The current time as a string.
        """
        current_time = datetime.now().strftime("%H:%M:%S")
        return f"Current time: {current_time}"

    def get_datetime(self) -> str:
        """
        Get the current date and time.

        :return: The current date and time as a string.
        """
        now = datetime.now()
        formatted = now.strftime("%A, %B %d, %Y at %H:%M:%S")
        return f"It is {formatted}"

    def get_time_in_timezone(self, timezone: str) -> str:
        """
        Get the current time in a specific timezone.

        :param timezone: Timezone name (e.g., 'Asia/Hong_Kong', 'America/New_York', 'Europe/London', 'UTC')
        :return: The current time in the specified timezone.
        """
        try:
            # Get current time in specified timezone
            tz = ZoneInfo(timezone)
            now = datetime.now(tz)

            # Format the time with timezone info
            formatted_time = now.strftime("%H:%M:%S")
            formatted_date = now.strftime("%A, %B %d, %Y")
            tz_abbr = now.strftime("%Z")  # Get timezone abbreviation (e.g., HKT, EST)

            return f"Current time in {timezone}: {formatted_time} {tz_abbr} on {formatted_date}"

        except Exception as e:
            # List of common timezones for help
            common_timezones = [
                "Asia/Hong_Kong (HKT)",
                "America/New_York (EST/EDT)",
                "America/Los_Angeles (PST/PDT)",
                "Europe/London (GMT/BST)",
                "Asia/Tokyo (JST)",
                "Asia/Shanghai (CST)",
                "Australia/Sydney (AEDT/AEST)",
                "UTC",
            ]
            return (
                f"Error: Invalid timezone '{timezone}'. "
                f"Common timezones: {', '.join(common_timezones)}. "
                f"Error details: {str(e)}"
            )
