from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score,confusion_matrix,classification_report
from fraud_detection.exception.exception import fraud_detection_exception
from fraud_detection.logger.logging import logging
from fraud_detection.entity.artifact_entity import ClassificationMetricArtifact
import sys



def get_classification_report(y_true,y_pred)->ClassificationMetricArtifact:
        try:
            f1_scores = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            logging.info(f"F1 Score: {f1_scores}, Precision: {precision}, Recall: {recall}")
            classification_report = ClassificationMetricArtifact(
                f1_score=f1_scores,
                precision_score=precision,
                recall_score=recall
            )
            return classification_report
        except Exception as e:
              raise fraud_detection_exception(e,sys) from e