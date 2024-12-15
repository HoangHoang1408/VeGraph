from langchain_core.output_parsers import JsonOutputParser

json_parser = JsonOutputParser()

class JSONParser:
    @staticmethod
    def extract_json_list(text):
        text = text.replace("'", "")
        first = text.find("[")
        last = text.rfind("]")
        return json_parser.parse(text[first:last+1])

    @staticmethod
    def extract_json_dict(text):
        text = text.replace("'", "")
        first = text.find("{")
        last = text.rfind("}")
        return json_parser.parse(text[first:last+1])