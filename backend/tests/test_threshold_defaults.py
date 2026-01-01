import uuid

from app.schemas import MultiSealComparisonRequest, MultiSealComparisonTaskCreate, PdfCompareRequest
from core.seal_compare import SealComparator


def test_default_threshold_is_0_5_in_seal_comparator():
    comp = SealComparator()
    assert comp.threshold == 0.5


def test_schema_default_threshold_is_0_5():
    dummy_ids = [uuid.uuid4()]

    req = MultiSealComparisonRequest(seal_image_ids=dummy_ids)
    assert req.threshold == 0.5

    task = MultiSealComparisonTaskCreate(
        task_uid="t",
        image1_id=uuid.uuid4(),
        seal_image_ids=dummy_ids,
    )
    assert task.threshold == 0.5

    pdf = PdfCompareRequest(
        image2_pdf_id=uuid.uuid4(),
    )
    assert pdf.threshold == 0.5


