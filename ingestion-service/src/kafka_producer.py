import json
import logging
from confluent_kafka import Producer
from src.config import settings

logger = logging.getLogger(__name__)

class DocumentProducer:
    def __init__(self, broker: str = settings.KAFKA_BROKER):
        self.producer = Producer({'bootstrap.servers': broker})
        self.topic = settings.KAFKA_TOPIC_RAW

    def delivery_report(self, err, msg):
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.info(f"Message delivered to {msg.topic()} [{msg.partition()}]")

    def send_document(self, doc_id: str, text: str, metadata: dict = None):
        if metadata is None:
            metadata = {}
            
        payload = {
            "doc_id": doc_id,
            "text": text,
            "metadata": metadata
        }
        
        try:
            self.producer.produce(
                self.topic,
                key=doc_id.encode('utf-8'),
                value=json.dumps(payload).encode('utf-8'),
                callback=self.delivery_report
            )
            self.producer.poll(0)
            return True
        except Exception as e:
            logger.error(f"Error producing message: {e}")
            return False

    def flush(self):
        self.producer.flush()

# Propósito general, instanciable o se puede usar un singleton
producer_instance = DocumentProducer()
