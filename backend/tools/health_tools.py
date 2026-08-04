"""Time, weather, and lightweight planning tools for the health agent."""

import json
from datetime import datetime
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from langchain.tools import tool

from common.tool_logging import log_tool_end, log_tool_start


WEATHER_LABELS = {
    0: "晴",
    1: "大致晴朗",
    2: "多云",
    3: "阴",
    45: "有雾",
    48: "有雾凇",
    51: "毛毛雨",
    53: "毛毛雨",
    55: "较强毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "阵雨",
    81: "阵雨",
    82: "强阵雨",
    95: "雷暴",
}

HEALTH_TOOL_NAMES = {
    "get_current_time",
    "get_today_weather",
    "generate_today_exercise_plan",
    "generate_weekly_diet_plan",
}


def _get_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "PersonalFitnessAssistant/0.1"})
    with urlopen(request, timeout=8) as response:  # nosec B310: fixed HTTPS endpoints only
        return json.loads(response.read().decode("utf-8"))


def fetch_today_weather(city: str) -> dict[str, Any]:
    """Fetch current and today's weather from Open-Meteo's public API."""
    if not city.strip():
        return {"error": "尚未设置城市，无法查询天气。"}
    try:
        query = urlencode({"name": city, "count": 1, "language": "zh", "format": "json"})
        location_data = _get_json(f"https://geocoding-api.open-meteo.com/v1/search?{query}")
        locations = location_data.get("results", [])
        if not locations:
            return {"error": f"未找到城市“{city}”，请在长期档案中更新 city。"}
        location = locations[0]
        weather_query = urlencode(
            {
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "current": "temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
                "timezone": "auto",
                "forecast_days": 1,
            }
        )
        weather_data = _get_json(f"https://api.open-meteo.com/v1/forecast?{weather_query}")
        current = weather_data["current"]
        daily = weather_data["daily"]
        return {
            "city": location["name"],
            "country": location.get("country"),
            "timezone": weather_data.get("timezone"),
            "date": daily["time"][0],
            "condition": WEATHER_LABELS.get(current["weather_code"], "未知天气"),
            "temperature_c": current["temperature_2m"],
            "apparent_temperature_c": current["apparent_temperature"],
            "wind_speed_kmh": current["wind_speed_10m"],
            "precipitation_mm": current["precipitation"],
            "high_c": daily["temperature_2m_max"][0],
            "low_c": daily["temperature_2m_min"][0],
            "rain_probability": daily["precipitation_probability_max"][0],
        }
    except Exception:
        return {"error": "天气服务暂时不可用，请改用室内、低风险的运动建议。"}


def _is_bad_outdoor_weather(weather: dict[str, Any]) -> bool:
    return (
        "error" in weather
        or weather.get("rain_probability", 0) >= 50
        or weather.get("wind_speed_kmh", 0) >= 35
        or weather.get("condition") in {"大雨", "强阵雨", "雷暴", "大雪"}
    )


def _diet_planning_brief(
    goal: str,
    restrictions: list[str],
    preferences: list[str],
    days: int,
    meals_per_day: int,
) -> dict[str, Any]:
    portion_rule = (
        "每餐优先保证一掌心优质蛋白、半盘非淀粉蔬菜、约一拳头主食；少油少糖。"
        if "减脂" in goal
        else "每餐保证蛋白质、蔬菜和主食齐全；根据饥饿感和运动量调整主食与加餐。"
    )
    return {
        "days": days,
        "meals_per_day": meals_per_day,
        "goal": goal,
        "must_avoid": restrictions,
        "preferences": preferences,
        "portion_rule": portion_rule,
        "protein_options": ["鸡蛋", "鸡胸肉", "鱼肉", "瘦牛肉", "豆腐", "无糖酸奶", "牛奶"],
        "carb_options": ["燕麦", "糙米", "杂粮饭", "红薯", "玉米", "全麦面包", "荞麦面"],
        "vegetable_options": ["西兰花", "菠菜", "番茄", "黄瓜", "生菜", "菌菇", "胡萝卜"],
        "fat_options": ["坚果", "牛油果", "橄榄油", "芝麻酱"],
        "generation_rules": [
            "根据用户本轮输入和长期档案生成菜单，不要逐字照抄候选食材。",
            "每天食材和烹饪方式尽量变化，避免 7 天高度重复。",
            "严格避开 must_avoid 中的忌口；不确定是否冲突时选择替代食材。",
            "输出普通生活方式建议，不做医学营养处方或精确热量处方。",
        ],
    }


def build_health_tools(profile: dict[str, Any]) -> list[Any]:
    """Bind stable, confirmed profile data to the tools for one agent invocation."""

    profile_city = str(profile.get("city") or "").strip()
    goal = str(profile.get("goal") or "健康、可持续生活方式")
    restrictions = [str(value) for value in profile.get("food_restrictions", [])]
    limitations = [str(value) for value in profile.get("exercise_limitations", [])]
    preferences = [str(value) for value in profile.get("preferences", [])]

    @tool
    def get_current_time(timezone_name: str = "Asia/Shanghai") -> dict[str, str]:
        """查询当前时间。用户询问现在几点、日期或安排时间时使用。"""
        log_tool_start("get_current_time", {"timezone_name": timezone_name})
        try:
            now = datetime.now(ZoneInfo(timezone_name))
        except Exception:
            result = {"error": f"不支持时区 {timezone_name}，请使用 IANA 时区名称。"}
            log_tool_end("get_current_time", result)
            return result
        result = {"timezone": timezone_name, "datetime": now.isoformat(timespec="seconds")}
        log_tool_end("get_current_time", result)
        return result

    @tool
    def get_today_weather(city: str | None = None) -> dict[str, Any]:
        """查询指定城市或长期健康档案 city 对应的当天实时天气。用户提供城市时传入 city；回答天气或户外运动前使用。"""
        log_tool_start("get_today_weather", {"city": city})
        selected_city = (city or profile_city).strip()
        if not selected_city:
            result = {"error": "用户未提供城市，且长期健康档案尚未设置 city，请先询问用户所在城市。"}
            log_tool_end("get_today_weather", result)
            return result
        result = fetch_today_weather(selected_city)
        log_tool_end("get_today_weather", result)
        return result

    @tool
    def generate_today_exercise_plan(
        available_minutes: int = 30, city: str | None = None
    ) -> dict[str, Any]:
        """根据个人目标、限制、偏好和指定城市当天的天气生成运动方案。用户提供城市时传入 city；用户请求运动方案时使用。"""
        log_tool_start(
            "generate_today_exercise_plan",
            {"available_minutes": available_minutes, "city": city},
        )
        minutes = min(max(available_minutes, 10), 120)
        selected_city = (city or profile_city).strip()
        weather = fetch_today_weather(selected_city) if selected_city else {"error": "未设置城市"}
        needs_low_impact = any(
            keyword in " ".join(limitations).lower()
            for keyword in ("膝", "踝", "腰", "疼", "伤", "knee", "ankle", "pain")
        )
        indoor = _is_bad_outdoor_weather(weather) or needs_low_impact
        if indoor:
            activity = "室内快走/原地踏步、靠墙静蹲（无疼痛时）、弹力带划船和温和拉伸"
        else:
            activity = "户外快走或轻松骑行，结束后进行温和拉伸"
        result = {
            "goal": goal,
            "weather": weather,
            "plan": f"{minutes} 分钟：5 分钟热身；{max(minutes - 10, 10)} 分钟{activity}；5 分钟放松。",
            "limitations_applied": limitations or ["无已确认限制"],
            "preferences_considered": preferences or ["无已确认偏好"],
            "safety": "任何动作出现疼痛、头晕或胸闷时立即停止；有旧伤请避免加重症状的动作。",
        }
        log_tool_end("generate_today_exercise_plan", result)
        return result

    @tool
    def generate_weekly_diet_plan(days: int = 7, meals_per_day: int = 3) -> dict[str, Any]:
        """生成一周饮食规划所需的个人约束、份量规则和候选食材。用户请求饮食计划时使用，并根据工具结果组织最终菜单。"""
        log_tool_start("generate_weekly_diet_plan", {"days": days, "meals_per_day": meals_per_day})
        safe_days = min(max(days, 1), 14)
        safe_meals_per_day = min(max(meals_per_day, 1), 5)
        result = {
            "planning_brief": _diet_planning_brief(
                goal=goal,
                restrictions=restrictions,
                preferences=preferences,
                days=safe_days,
                meals_per_day=safe_meals_per_day,
            ),
            "note": "这是一般生活方式建议；孕期、慢病、进食障碍或需医学营养治疗时，请咨询医生或注册营养师。",
        }
        log_tool_end("generate_weekly_diet_plan", result)
        return result

    return [
        get_current_time,
        get_today_weather,
        generate_today_exercise_plan,
        generate_weekly_diet_plan,
    ]
