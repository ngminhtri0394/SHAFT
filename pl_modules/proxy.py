import matgl
import numpy as np

class M3gnetDGL_Proxy():
    def __init__(self, model_name, max_atom=100) -> None:
        self.model_name = model_name
        self.model = matgl.load_model(model_name)
        self.max_atom = 100

    def __call__(self, m):
        small_crystal_mask = np.asarray([(len(i.structure.species)<self.max_atom) and (i.structure.density > 1.5) and (i.structure.density < 5.0) for i in m])
        m_small = m[small_crystal_mask]
        preds = np.full(len(m),10.0)
        if len(m_small) == 0:
            return preds
        pred = [float(self.model.predict_structure(struct.structure).numpy()) for struct in m_small]
        # preds = [float(self.model.predict_structure(struct.structure).numpy()) for struct in m]
        preds[small_crystal_mask] = pred
        # preds[preds < -5] = 10
        preds = np.nan_to_num(preds,nan=10.0)
        preds = np.clip(preds, a_min=-10.0, a_max=10.0)
        return preds


    