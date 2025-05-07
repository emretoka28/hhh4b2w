# coding: utf-8

"""
Test model definition.
"""

from __future__ import annotations

import law
import order as od
import pickle
import re

from columnflow.types import Any
from columnflow.ml import MLModel
from columnflow.util import maybe_import, dev_sandbox
from columnflow.columnar_util import Route, set_ak_column, remove_ak_column

ak = maybe_import("awkward")
tf = maybe_import("tensorflow")
np = maybe_import("numpy")
keras = maybe_import("tensorflow.keras")

law.contrib.load("tensorflow")
logger = law.logger.get_logger(__name__)

class DNNModel(MLModel):

    # mark the model as accepting only a single config
    single_config = True
    input_features_namespace = "MLInput"


    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # prepend namespace to input features
        self.input_columns = {
            f"{self.input_features_namespace}.{name}"
            for name in self.input_features
        }

    # -- methods related to task setup & environment

    def setup(self):
        # dynamically add variables for the quantities produced by this model
        if f"{self.cls_name}.output" not in self.config_inst.variables:
            self.config_inst.add_variable(
                name=f"{self.cls_name}.output",
                null_value=-1,
                binning=(20, 0, 1.0),
                x_title=f"DNN output",
            )

    def sandbox(self, task: law.Task) -> str:
        # return dev_sandbox("bash::$AZH_BASE/sandboxes/venv_ml.sh")
        return dev_sandbox("bash::$HHH4B2W_BASE/sandboxes/venv_ml.sh")

    def datasets(self, config_inst: od.Config) -> set[od.Dataset]:
        return {
            config_inst.get_dataset("hhh_bbbbww_c3_0_d4_0_amcatnlo"),
            # config_inst.get_dataset("hhh_bbbbww_c3_0_d4_99_amcatnlo"),
            config_inst.get_dataset("hhh_bbbbww_c3_0_d4_m1_amcatnlo"),
            config_inst.get_dataset("hhh_bbbbww_c3_19_d4_19_amcatnlo"),
            # config_inst.get_dataset("hhh_bbbbww_c3_1_d4_0_amcatnlo"),
            # config_inst.get_dataset("hhh_bbbbww_c3_1_d4_2_amcatnlo"),
            # config_inst.get_dataset("hhh_bbbbww_c3_2_d4_m1_amcatnlo"),
            # config_inst.get_dataset("hhh_bbbbww_c3_4_d4_9_amcatnlo"),
            config_inst.get_dataset("hhh_bbbbww_c3_m1_d4_0_amcatnlo"),
            config_inst.get_dataset("hhh_bbbbww_c3_m1_d4_m1_amcatnlo"),
            

            config_inst.get_dataset("tt_sl_powheg"),
            config_inst.get_dataset("tt_dl_powheg"),
            config_inst.get_dataset("tt_fh_powheg"),
            config_inst.get_dataset("hh_ggf_hbb_hvv_kl1_kt1_powheg"),
            config_inst.get_dataset("tth_hbb_powheg"),

        }

    def uses(self, config_inst: od.Config) -> set[Route | str]:
        return {"normalization_weight"} | self.input_columns
        return self.input_columns

    def produces(self, config_inst: od.Config) -> set[Route | str]:
        return {
            f"{self.cls_name}.output",
        }

    def output(self, task: law.Task) -> law.FileSystemDirectoryTarget:
        return task.target(f"mlmodel_f{task.branch}of{self.folds}", dir=True)

    def open_model(self, target: law.FileSystemDirectoryTarget) -> tf.keras.models.Model:
        return target.load(formatter="tf_keras_model")


    def prepare_inputs(
        self,
        task,
        input,
    ) -> tuple[dict[str, np.array]]:

        # obtain processes from config
        process_insts = [
            self.config_inst.get_process(proc)
            for proc in self.processes
        ]

        proc_n_events = np.array(len(self.processes) * [0])
        proc_custom_weights = np.array(len(self.processes) * [0])
        proc_sum_weights = np.array(len(self.processes) * [0.0])
        proc_idx = {}  # bookkeeping which process each dataset belongs to

        #
        # determine process of each dataset and count number of events & sum of eventweights for this process
        #

        for dataset, files in input["events"][self.config_inst.name].items():
            dataset_inst = self.config_inst.get_dataset(dataset)

            # check dataset only belongs to one process
            if len(dataset_inst.processes) != 1:
                raise Exception("only 1 process inst is expected for each dataset")

            # TODO: use stats here instead
            # mlevents = [
            #     ak.from_parquet(inp["mlevents"].fn)
            #     for inp in files
            # ]
            mlevents= ak.Array([])
            #slimming to signal catgory
            for inp in files:
                events = ak.from_parquet(inp["mlevents"].path)

                if len(mlevents)==0:
                    mlevents = [events]
                else:
                    mlevents = ak.concatenate(mlevents2,events)
            # from IPython import embed; embed()
            n_events = sum(
                len(events)
                for events in mlevents
            )
            # from  on import embed; embed()
            sum_weights = sum(
                ak.sum(events.normalization_weight)
                for events in mlevents
            )
            # from IPython import embed; embed()

            for i, proc in enumerate(process_insts):
                # print("LEAF")
                # print(proc)
                proc_custom_weights[i] = self.proc_custom_weights[proc.name]
                leaf_procs = [
                    p for p, _, _ in self.config_inst.get_process(proc).walk_processes(include_self=True)
                ]
                # print(proc)
                # print(leaf_procs)
                # print(dataset_inst.processes.get_first())
                # from IPython import embed; embed()
                if dataset_inst.processes.get_first() in leaf_procs:
                # if True:
                    logger.info(f"the dataset *{dataset}* is used for training the *{proc.name}* output node")
                    proc_idx[dataset] = i
                    proc_n_events[i] += n_events
                    proc_sum_weights[i] += sum_weights
                    # if proc_sum_weights[i] == 0:
                    #     proc_sum_weights[i] = 1
                    # from IPython import embed; embed()
                    continue
                # from IPython import embed; embed()
            # print("Leaving LEAF")

        # fail if no process was found for dataset
            if proc_idx.get(dataset) is None:
                raise Exception(f"dataset {dataset} is not matched to any of the given processes")


        DNN_inputs = {
            "weights": None,
            "inputs": None,
            "target": None,
        }

        # scaler for weights such that the largest are of order 1
        weights_scaler = min(proc_n_events / proc_custom_weights)
        # from IPython import embed; embed()c
        sum_nnweights_processes = {}
        for dataset, files in input["events"][self.config_inst.name].items():
            # print(dataset)
            # print(proc_idx)
            this_proc_idx = proc_idx[dataset]
            # print(this_proc_idx)

            this_proc_name = self.processes[this_proc_idx]
            this_proc_n_events = proc_n_events[this_proc_idx]
            this_proc_sum_weights = proc_sum_weights[this_proc_idx]
            
            # from IPython import embed; embed()
            logger.info(
                f"dataset: {dataset}, \n"
                f"  #Events: {this_proc_n_events}, \n"
                f"  Sum Eventweights: {this_proc_sum_weights}",
            )
            sum_nnweights = 0

            for inp in files:
                events = ak.from_parquet(inp["mlevents"].path)
                # padded_cats = np.array([np.pad(arr,(0,max(2-len(arr),0)),constant_values=None) for arr in events.category_ids],dtype=object)
                #lens = np.array([len(x) for x in events.category_ids])
                #events = events[lens>1]
                #signal = np.bitwise_and(events.category_ids[:,1]%10000>1900, events.category_ids[:,1]%1000<200)
                #events = events[signal]
                weights = events.normalization_weight
                # from IPython import embed; embed()

                if self.eqweight:
                    weights = weights * weights_scaler / this_proc_sum_weights
                    custom_procweight = self.proc_custom_weights[this_proc_name]
                    weights = weights * custom_procweight

                weights = ak.to_numpy(weights)
                # from IPython import embed; embed()
                if np.any(~np.isfinite(weights)):
                    raise Exception(f"Non-finite values found in weights from dataset {dataset}")

                sum_nnweights += sum(weights)
                sum_nnweights_processes.setdefault(this_proc_name, 0)
                sum_nnweights_processes[this_proc_name] += sum(weights)
                
                # remove columns not used in training
                input_features = events[self.input_features_namespace]
                for var in input_features.fields:
                    # print(var)
                    if var not in self.input_features:
                        events = remove_ak_column(events, f"{self.input_features_namespace}.{var}")
                # from IPython import embed; embed()
                        
                
                #add loop to check if none values are in events.MLInputs.fields
                # for field in events.MLInput.fields:
                #     values = getattr(events.MLInput, field)
                #     print("Field:", field)
                #     print(values)
                #     print(type(values))
                #     print("")
                # from IPython import embed; embed()
                ml_inputs = events.MLInput
                # from IPython import embed; embed()
                ml_inputs = ak.to_numpy(ml_inputs)
                # from IPython import embed; embed()
                # events = ak.to_numpy(events)
                
                # events = events.astype(
                #     [(name, np.float32) for name in events.dtype.names],
                #     copy=False,
                # ).view(np.float32).reshape((-1, len(events.dtype)))

                ml_inputs = ml_inputs.astype(
                    [(name, np.float32) for name in ml_inputs.dtype.names],
                    copy=False,
                ).view(np.float32).reshape((-1, len(ml_inputs.dtype)))

                if np.any(~np.isfinite(ml_inputs)):
                    raise Exception(f"Non-finite values found in weights from dataset {dataset}")
                # create the truth values for the output layer
                # target = np.zeros((len(events), len(self.processes)))
                # target[:, this_proc_idx] = 1
                # print("Setting target")
                # print(dataset)
                # print(self.config_inst.get_dataset(dataset))
                # print(self.config_inst.get_dataset(dataset).has_tag("is_ttbar"))
                # print(self.config_inst.get_dataset(dataset).has_tag("is_signal"))

                target = np.ones((len(ml_inputs), 1)) if self.config_inst.get_dataset(dataset).has_tag("hhh4b2w") else np.zeros((len(ml_inputs), 1))
                # target = np.ones((len(events), 1)) if self.config_inst.get_dataset(dataset).has_tag("is_signal") else np.zeros((len(events), 1))
                
                # target = np.ones((len(events), 1))

                if np.any(~np.isfinite(target)):
                    raise Exception(f"Non-finite values found in target from dataset {dataset}")
                # print("Concatenate!")
                # # print(DNN_inputs)
                # print("DNN",DNN_inputs["inputs"])
                # if DNN_inputs["inputs"] is not None:
                #     print("DNN len",len(DNN_inputs["inputs"]))
                #     print("DNN shape",ak.num(DNN_inputs["inputs"],axis=0))
                #     print("DNN shape",ak.num(DNN_inputs["inputs"],axis=1))
                # print("events",events)
                # print("len events",len(events))
                # print("events shape",ak.num(events,axis=0))
                # print("events shape",ak.num(events,axis=1))
                if DNN_inputs["weights"] is None:
                    DNN_inputs["weights"] = weights
                    DNN_inputs["inputs"] = ml_inputs
                    # DNN_inputs["inputs"] = events
                    DNN_inputs["target"] = target
                else:
                    DNN_inputs["weights"] = np.concatenate([DNN_inputs["weights"], weights])
                    # DNN_inputs["inputs"] = np.concatenate([DNN_inputs["inputs"], events])
                    DNN_inputs["inputs"] = np.concatenate([DNN_inputs["inputs"], ml_inputs])
                    DNN_inputs["target"] = np.concatenate([DNN_inputs["target"], target])
                # from IPython import embed; embed()
        #
        # shuffle events and split into train and validation part
        #
        inputs_size = sum([arr.size * arr.itemsize for arr in DNN_inputs.values()])
        logger.info(f"inputs size is {inputs_size / 1024**3} GB")

        shuffle_indices = np.array(range(len(DNN_inputs["weights"])))
        np.random.shuffle(shuffle_indices)

        n_validation_events = int(self.validation_fraction * len(DNN_inputs["weights"]))

        train, validation = {}, {}
        for k in DNN_inputs.keys():
            DNN_inputs[k] = DNN_inputs[k][shuffle_indices]

            validation[k] = DNN_inputs[k][:n_validation_events]
            train[k] = DNN_inputs[k][n_validation_events:]
        
        return train, validation

    def train(
        self,
        task: law.Task,
        input: dict[str, list[dict[str, law.FileSystemFileTarget]]],
        output: law.FileSystemDirectoryTarget,
    ) -> None:
        # from IPython import embed; embed()
        # define a dummy NN

            # run on GPU
        gpus = tf.config.list_physical_devices("GPU")

        # restrict to run only on first GPU
        # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            # GPU already initialized -> print warning and continue
            print(e)
        except IndexError:
            print("No GPUs found. Will use CPU.")

        #
        # prepare input
        #

        # TODO: implement
        print("IN THE TRAINING!")
        print(task)
        print(input)
        print("Output from prepare inputs:")
        print(self.prepare_inputs(task, input))
        train, validation = self.prepare_inputs(task, input)

        # check for non-finite values (inf, nan)
        for key in train.keys():
            if np.any(~np.isfinite(train[key])):
                raise Exception(f"Non-finite values found in training {key}")
            if np.any(~np.isfinite(validation[key])):
                raise Exception(f"Non-finite values found in validation {key}")

        #
        # prepare model
        #

        # TODO: implement
        n_inputs = len(self.input_features)
        # from IPython import embed; embed()
        # n_outputs = len(self.processes)
        n_outputs = 1

        # start model definition
        model = keras.Sequential()

        # define input normalization
        model.add(keras.layers.BatchNormalization(input_shape=(n_inputs,)))

        # hidden layers
        for n_nodes in self.layers:
            model.add(keras.layers.Dense(
                units=n_nodes,
                activation="ReLU",
            ))

            # optional dropout after each hidden layer
            if self.dropout:
                model.add(keras.layers.Dropout(self.dropout))

        # output layer
        model.add(keras.layers.Dense(
            n_outputs,
            activation="sigmoid",
        ))

        # optimizer
        # settings from https://github.com/jabuschh/ZprimeClassifier/blob/8c3a8eee/Training.py#L93  # noqa
        optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9, beta_2=0.999,
            epsilon=1e-6,
            amsgrad=False,
        )

        # compile model
        model.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            # metrics = ["accuracy"],
            weighted_metrics=["binary_accuracy","accuracy"],
        )

        #
        # training
        #

        # early stopping criteria
        early_stopping = keras.callbacks.EarlyStopping(
            # stop when validation loss no longer improves
            monitor="val_loss",
            mode="min",
            # minimum change to consider as improvement
            min_delta=0.005,
            # wait this many epochs w/o improvement before stopping
            patience=max(1, int(self.epochs / 4)),  # 100
            # start monitoring from the beginning
            start_from_epoch=0,
            verbose=0,
            restore_best_weights=True,
        )

        # learning rate reduction on plateau
        lr_reducer = keras.callbacks.ReduceLROnPlateau(
            # reduce LR when validation loss stops improving
            monitor="val_loss",
            mode="min",
            # minimum change to consider as improvement
            min_delta=0.001,
            # factor by which the learning rate will be reduced
            factor=0.5,
            # wait this many epochs w/o improvement before reducing LR
            patience=max(1, int(self.epochs / 8)),  # 100
        )

        # construct TF datasets
        with tf.device("CPU"):
            # training
            tf_train = tf.data.Dataset.from_tensor_slices(
                (train["inputs"], train["target"], train["weights"]),
            ).batch(self.batchsize)

            # validation
            tf_validate = tf.data.Dataset.from_tensor_slices(
                (validation["inputs"], validation["target"], validation["weights"]),
            ).batch(self.batchsize)

        # do training
        model.fit(
            tf_train,
            validation_data=tf_validate,
            epochs=self.epochs,
            callbacks=[early_stopping, lr_reducer],
            verbose=2,
        )

        # save trained model and history
        output.parent.touch()
        model.save(output.path)
        with open(f"{output.path}/model_history.pkl", "wb") as f:
            pickle.dump(model.history.history, f)


    def evaluate(
        self,
        task: law.Task,
        events: ak.Array,
        models: list[Any],
        fold_indices: ak.Array,
        events_used_in_training: bool = False,
    ) -> ak.Array:

        inputs = ak.copy(events)
        # from IPython import embed; embed()
        input_features = inputs[self.input_features_namespace]
        for var in input_features.fields:
            if var not in self.input_features:
                inputs = remove_ak_column(inputs, f"{self.input_features_namespace}.{var}")
        inputs = inputs[self.input_features_namespace]
        # from IPython import embed; embed()
        inputs = ak.to_numpy(inputs)
        inputs = inputs.astype(
            [(name, np.float32) for name in inputs.dtype.names],
            copy=False,
        ).view(np.float32).reshape((-1, len(inputs.dtype)))
        # from IPython import embed; embed()
        predictions = []
        for i, model in enumerate(models):
            prediction = ak.from_numpy(model.predict_on_batch(inputs))
            predictions.append(prediction)
        outputs = ak.ones_like(predictions[0]) * -1
        # from IPython import embed; embed()
        for i in range(self.folds):
            # reshape mask from N*bool to N*k*bool (TODO: simpler way?)
            idx = ak.to_regular(
                ak.concatenate(
                    [
                        ak.singletons(fold_indices == i),
                    ] ,
                    axis=1,
                ),
            )
            outputs = ak.where(idx, predictions[i], outputs)
        # from IPython import embed; embed()
        events = set_ak_column(events, f"{self.cls_name}.output", outputs)
        return events


# usable derivations
DNN = DNNModel.derive("DNN_v2", cls_dict={
    "batchsize": 500,
    "dropout": 0.5,
    "epochs": 500,
    "folds": 2,
    "eqweight": True,
    "validation_fraction": 0.25,
    "layers": [512, 512],
    "learning_rate": 0.0005,


    "processes": [
        #background
        "tt",
        "tth",
        "hh_ggf",
        #signal
        "hhh_bbbbww_c3_0_d4_0",
        # "hhh_bbbbww_c3_0_d4_99",
        "hhh_bbbbww_c3_0_d4_m1",
        "hhh_bbbbww_c3_19_d4_19",
        # "hhh_bbbbww_c3_1_d4_0",
        # "hhh_bbbbww_c3_1_d4_2",
        # "hhh_bbbbww_c3_2_d4_m1",
        # "hhh_bbbbww_c3_4_d4_9",
        "hhh_bbbbww_c3_m1_d4_0",
        "hhh_bbbbww_c3_m1_d4_m1",
    ],

    "proc_custom_weights": {
        "tt": 5,
        "tth": 5,
        "hh_ggf": 0.5,
        "hhh_bbbbww_c3_0_d4_0": 1,
        # "hhh_bbbbww_c3_0_d4_99": 1,
        "hhh_bbbbww_c3_0_d4_m1": 1,
        "hhh_bbbbww_c3_19_d4_19": 1,
        # "hhh_bbbbww_c3_1_d4_0": 1,
        # "hhh_bbbbww_c3_1_d4_2": 1,
        # "hhh_bbbbww_c3_2_d4_m1": 1,
        # "hhh_bbbbww_c3_4_d4_9": 1,
        "hhh_bbbbww_c3_m1_d4_0": 1,
        "hhh_bbbbww_c3_m1_d4_m1": 1,
    },

    "input_features": [
    ] + [
        f"Jet{i + 1}_{var}"
        for var in ("pt", "eta", "phi", "mass")
        for i in range(5)
    ] + [
        f"BJet{i+1}_{var}"
        for var in ("pt", "eta", "phi", "mass")
        for i in range(3)

    ] +
    #  +[
    #     f"lepton_{var}"
    #     for var in ("pt", "eta", "phi" )
    # ] +
    #   [
    #     f"Electron_{var}"
    #     for var in ("pt", "eta", "phi")
    # ]+ [
    #     f"Muon_{var}"
    #     for var in ("pt", "eta", "phi")
    # ]+ 
    [   "n_jet",
        "n_bjet",
        "m_jj",
        "jj_pt",
        "deltaR_jj",
        "dr_min_jj",
        "dr_mean_jj",
        "m_bb",
        "bb_pt",
        "deltaR_bb",
        "dr_min_bb",
        "dr_mean_bb",
        "deltaR_lbb",
    ]
    
    })
