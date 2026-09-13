"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import re
from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""


class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""
    def query(self, user_input: str) -> str:
        answer = f"Tôi đang hỗ trợ bạn về: {user_input}. Tôi chưa dùng tool nào và chỉ trả lời bằng logic đơn giản."
        return {
            "status": "success",
            "tool_calls": [],
            "answer": answer,
        }


class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace = []

    def _parse_price(self, text: str):
        lower_text = text.lower()
        match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:triệu|trieu|m\b)", lower_text)
        if not match:
            return 5000000
        value = float(match.group(1).replace(",", "."))
        return int(value * 1_000_000)

    def _extract_trip_codes(self, text: str):
        valid_codes = {"HAN", "SGN", "DAD"}
        lower_text = text.lower()

        patterns = [
            r"(?:từ|from)\s*([A-Z]{3})\s*(?:đi|to|đến)\s*([A-Z]{3})",
            r"([A-Z]{3})\s*(?:đi|to|đến)\s*([A-Z]{3})",
            r"(?:từ|from)\s*([A-Z]{3})\s*(?:đi|to|đến)\s*([a-z]{3})",
            r"([A-Z]{3})\s*(?:đi|to|đến)\s*([a-z]{3})",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                origin, destination = match.group(1).upper(), match.group(2).upper()
                if origin in valid_codes and destination in valid_codes:
                    return {"origin": origin, "destination": destination}

        matches = [m.upper() for m in re.findall(r"\b[A-Z]{3}\b", text.upper()) if m.upper() in valid_codes]
        if len(matches) >= 2:
            return {"origin": matches[0], "destination": matches[1]}

        return {"origin": "HAN", "destination": "SGN"}

    def _extract_city_for_weather(self, text: str, fallback: str = "SGN"):
        valid_codes = {"HAN", "SGN", "DAD"}
        lower = text.lower()

        explicit_patterns = [
            r"thời\s+tiết\s+(?:ở|tại)?\s*([A-Z]{3})",
            r"(?:ở|tại)\s*([A-Z]{3})\s*(?:hiện\s+tại|nên\s+mặc\s+gì|mặc\s+gì)",
            r"(?:đà\s+nẵng|danang|dà\s+nẵng)",
            r"(?:hà\s+nội|hanoi)",
            r"(?:hồ\s+chí\s+minh|tp\.\s*hồ\s+chí\s+minh|saigon|sgn)",
        ]

        for pattern in explicit_patterns:
            match = re.search(pattern, lower, flags=re.IGNORECASE)
            if match:
                if pattern.startswith("(?:đà") or "đà nẵng" in match.group(0).lower():
                    return "DAD"
                if pattern.startswith("(?:hà") or "hà nội" in match.group(0).lower():
                    return "HAN"
                if pattern.startswith("(?:hồ") or "hồ chí minh" in match.group(0).lower() or "saigon" in match.group(0).lower() or "sgn" in match.group(0).lower():
                    return "SGN"
                value = match.group(1).upper() if match.lastindex else None
                if value in valid_codes:
                    return value

        matches = [m.upper() for m in re.findall(r"\b[A-Z]{3}\b", text.upper()) if m.upper() in valid_codes]
        if matches:
            if fallback in valid_codes and fallback in matches:
                return fallback
            return matches[0]
        return fallback

    def _format_flight_answer(self, flights, origin, destination, max_price):
        if not flights:
            return f"Không tìm thấy chuyến bay từ {origin} đi {destination} dưới {max_price:,} VND."

        lines = [f"Các chuyến bay phù hợp từ {origin} đi {destination} dưới {max_price:,} VND:"]
        for item in flights:
            lines.append(
                f"- {item['flight_number']} ({item['airline']}), khởi hành {item['departure_time']}, giá {item['price_vnd']:,} VND."
            )
        return "\n".join(lines)

    def _format_weather_answer(self, weather):
        if not weather or "error" in weather:
            return "Hiện chưa có dữ liệu thời tiết cho khu vực được yêu cầu."
        city = weather.get("city", "Thành phố")
        temp = weather.get("temperature_c", "N/A")
        condition = weather.get("condition", "N/A")
        recommendation = weather.get("recommendation", "")
        return f"Thời tiết tại {city}: {temp}°C, {condition}. {recommendation}"

    def run(self, user_input: str) -> str:
        self.trace = []
        text = user_input.strip()
        lower_text = text.lower()

        flight_keywords = ["chuyến bay", "vé", "flight", "máy bay", "đi", "đến", "từ"]
        weather_keywords = ["thời tiết", "weather", "mặc gì", "nhiệt độ", "dự báo", "mưa", "nắng"]
        faq_keywords = ["chính sách", "đổi trả", "vinpearl"]

        asks_flight = any(keyword in lower_text for keyword in flight_keywords)
        asks_weather = any(keyword in lower_text for keyword in weather_keywords)
        asks_faq = any(keyword in lower_text for keyword in faq_keywords)

        if asks_faq:
            answer = "Chính sách đổi trả vé máy bay Vinpearl thường cho phép đổi vé theo điều kiện của từng loại vé, và bạn nên kiểm tra chi tiết trên đơn đặt chỗ hoặc liên hệ tổng đài hỗ trợ."
            self.trace.append({
                "step": 1,
                "thought": "Câu hỏi là FAQ, không cần gọi tool.",
                "final_answer": answer,
            })
            return {"status": "completed", "iterations": 1, "answer": answer, "trace": self.trace}

        if not asks_flight and not asks_weather:
            answer = "Tôi chưa có đủ thông tin để trả lời chính xác. Bạn có thể hỏi về chuyến bay hoặc thời tiết theo mã thành phố như HAN, SGN, DAD."
            self.trace.append({
                "step": 1,
                "thought": "Không xác định được nhu cầu cần dùng tool.",
                "final_answer": answer,
            })
            return {"status": "completed", "iterations": 1, "answer": answer, "trace": self.trace}

        flight_data = {"origin": "HAN", "destination": "SGN"}
        if asks_flight:
            flight_data = self._extract_trip_codes(text)

        max_price = self._parse_price(text)
        weather_city = self._extract_city_for_weather(text, flight_data.get("destination", "SGN"))

        iteration = 1
        flight_done = False
        weather_done = False

        while iteration <= self.max_iterations:
            if asks_flight and not flight_done:
                action = {
                    "name": "get_flight_info",
                    "args": {
                        "origin": flight_data["origin"],
                        "destination": flight_data["destination"],
                        "max_price": max_price,
                    },
                }
                flights = get_flight_info(**action["args"])
                self.trace.append({
                    "step": iteration,
                    "thought": "Tôi cần tra cứu chuyến bay phù hợp theo điểm đi, điểm đến và ngân sách.",
                    "action": action,
                    "observation": flights,
                })
                flight_done = True

                if not asks_weather:
                    answer = self._format_flight_answer(flights, flight_data["origin"], flight_data["destination"], max_price)
                    return {"status": "completed", "iterations": len(self.trace), "answer": answer, "trace": self.trace}

                if iteration >= self.max_iterations:
                    return {
                        "status": "max_iterations_reached",
                        "iterations": len(self.trace),
                        "answer": "Không thể hoàn thành trong số bước tối đa.",
                        "trace": self.trace,
                    }

                iteration += 1
                continue

            if asks_weather and not weather_done:
                action = {"name": "get_weather_forecast", "args": {"city_code": weather_city}}
                weather = get_weather_forecast(**action["args"])
                self.trace.append({
                    "step": iteration,
                    "thought": "Tôi cần kiểm tra thời tiết của thành phố để đưa gợi ý mặc quần áo phù hợp.",
                    "action": action,
                    "observation": weather,
                })
                weather_done = True

                if flight_done:
                    if iteration >= self.max_iterations:
                        return {
                            "status": "max_iterations_reached",
                            "iterations": len(self.trace),
                            "answer": "Không thể hoàn thành trong số bước tối đa.",
                            "trace": self.trace,
                        }
                    final_answer = (
                        self._format_flight_answer(
                            get_flight_info(origin=flight_data["origin"], destination=flight_data["destination"], max_price=max_price),
                            flight_data["origin"],
                            flight_data["destination"],
                            max_price,
                        )
                        + "\n\n"
                        + self._format_weather_answer(weather)
                    )
                    self.trace.append({
                        "step": iteration + 1,
                        "thought": "Đã có đủ thông tin từ cả chuyến bay và thời tiết, tôi tổng hợp câu trả lời cuối cùng.",
                        "final_answer": final_answer,
                    })
                    return {"status": "completed", "iterations": len(self.trace), "answer": final_answer, "trace": self.trace}

                answer = self._format_weather_answer(weather)
                return {"status": "completed", "iterations": len(self.trace), "answer": answer, "trace": self.trace}

            if asks_flight and flight_done and asks_weather and weather_done:
                break

            iteration += 1

        answer = "Tôi đã tổng hợp thông tin nhưng chưa hoàn tất đủ dữ liệu để trả lời."
        return {"status": "completed", "iterations": len(self.trace), "answer": answer, "trace": self.trace}


def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result)
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()