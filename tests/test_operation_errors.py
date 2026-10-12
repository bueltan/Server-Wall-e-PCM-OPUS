import unittest
from bridge.operation_errors import already_absent, no_active_response

class OperationErrorTests(unittest.TestCase):
    def test_generic_rejections_never_mean_success(self):
        for message in ('', 'Invalid response_id parameter', 'Item deletion not supported',
                        'Invalid API key', 'Response cancellation not supported'):
            error={'code':'invalid_request_error','message':message}
            self.assertFalse(already_absent(error));self.assertFalse(no_active_response(error))

    def test_explicit_idempotent_outcomes(self):
        self.assertTrue(already_absent({'message':'Item abc not found'}))
        self.assertTrue(already_absent({'code':'conversation_item_not_found'}))
        self.assertTrue(no_active_response({'message':'No active response to cancel'}))
        self.assertTrue(no_active_response({'message':'Response abc already cancelled'}))
        self.assertFalse(no_active_response({'message':'Item abc not found'}))
        self.assertFalse(already_absent({'message':'No active response to cancel'}))
