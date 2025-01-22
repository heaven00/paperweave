# from langflow.field_typing import Data
from langflow.custom import Component
from langflow.io import StrInput, Output
from langflow.schema import Data
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from docling.document_converter import DocumentConverter


class DoclingComponent(Component):
    display_name = "Docling Component"
    description = "Convert url to heirarichal chunks of documents"
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "docling_component"
    name = "DoclingComponent"
    converter = DocumentConverter()
    chunker = HierarchicalChunker()


    inputs = [
        StrInput(name="url", display_name="URL"),
    ]

    outputs = [
        Output(display_name="Chunked Text", name="chunks", method="build_output"),
    ]

    def build_output(self) -> list[str]:
        doc = self.converter.convert(self.url).document
        return list(self.chunker.chunk(doc))

