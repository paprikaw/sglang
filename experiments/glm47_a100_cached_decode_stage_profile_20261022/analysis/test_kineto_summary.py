"""Synthetic CPU tests: none of these fixtures are experimental evidence."""
import importlib.util
from pathlib import Path
import tempfile
import json
import unittest

SPEC = importlib.util.spec_from_file_location("kineto_summary", Path(__file__).with_name("kineto_summary.py"))
K = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(K)


def event(name, start, duration, category="user_annotation", thread=1, process=10, **arguments):
    return dict(name=name, ts=start, dur=duration, cat=category, ph="X", tid=thread,
                pid=process, args=arguments)


def gpu(name, start, duration, correlation=None, category="kernel", device=0):
    args = dict(device=device)
    if correlation is not None:
        args["correlation"] = correlation
    return event(name, start, duration, category, thread=7, process=device, **args)


def api(index, parent, category, start, end, layer="3", label="module", stream=99):
    return dict(index=index, parent=parent, category=category, label=label,
                layer=layer, forward_index=1, extend_index=1, start_stream=stream,
                end_stream=stream, start_ms=start, end_ms=end, duration_ms=end-start,
                cpu_start_ns=index * 100)


class TestKinetoSummary(unittest.TestCase):
    def test_post_cohort_scheduler_kernel_is_not_model_gpu_work(self):
        events=[event("attr::model_forward::runner",0,10),
                event("attr::attention_full::3",1,8),
                event("cudaLaunchKernel",2,1,"cuda_runtime",correlation=7),
                gpu("model_kernel",4,8,7),
                event("attr::scheduler_get_next_batch::Scheduler",20,100),
                event("cudaLaunchKernel",21,1,"cuda_runtime",correlation=8),
                gpu("opaque_collective",23,90,8)]
        sample=dict(events=[api(0,None,"model_forward",0,12,layer=None)],cuda_device=0)
        out=K.summarize_trace(dict(traceEvents=events),sample)["gpu_devices"]["0"]
        self.assertEqual(out["activity_envelope"]["envelope_us"],109)
        # Launch membership retains asynchronous completion after CPU scope end.
        self.assertEqual(out["model_forward_gpu_work"]["activity"]["union_us"],8)
        self.assertEqual(out["model_forward_gpu_work"]["activity"]["end_us"],12)
        excluded=out["model_forward_gpu_work"]["excluded_top_duration_events"][0]
        self.assertEqual(excluded["cpu_scope_lineage"],[('scheduler_get_next_batch','Scheduler')])

    def test_exact_name_calibration_requires_two_training_and_independent_holdout(self):
        sample = dict(forwards=[dict(forward_index=i,extend_index=i,detailed=True)
                               for i in (2,8,14)])
        def item(forward,category,name="opaque",device="0"):
            start=forward if forward is not None else 20
            return dict(kind="kernel",category=category,device=device,name=name,
                        forward_index=forward,interval=(start,start+1))
        rows = [item(i,"deep_ep.dispatch") for i in (2,8,14)] + [item(4,"unknown")]
        out = K.calibrated_name_view(rows,sample)
        self.assertEqual(out["eligible_name_count"],1)
        self.assertEqual(out["classification_counts"]["calibrated_exact_name"],1)
        conflict = K.calibrated_name_view(rows+[item(3,"tp_collective")],sample)
        self.assertEqual(conflict["eligible_name_count"],0)
        missing = K.calibrated_name_view(rows[:2]+rows[3:],sample)
        self.assertEqual(missing["eligible_name_count"],0)
        # A different GPU, model-external activity, or copy is not extrapolated.
        rows += [item(5,"unknown",device="1"),item(None,"unknown")]
        rows[-1]["interval"]=(20,21)
        copy=item(6,"unknown");copy["kind"]="memcpy";rows.append(copy)
        out=K.calibrated_name_view(rows,sample)
        self.assertEqual(out["classification_counts"]["calibrated_exact_name"],1)

    def test_nested_runtime_correlations_override_kernel_name_guesses(self):
        events = [event("attr::model_forward::runner", 0, 100),
                  event("attr::layer::3", 1, 90),
                  event("attr::attention_full::3", 2, 40),
                  event("attr::humming_gemm::linear:opaque", 3, 8),
                  event("cudaLaunchKernel", 4, 1, "cuda_runtime", correlation=71),
                  gpu("misleading_expert_name", 5, 10, 71),
                  event("attr::deep_ep.dispatch_recv::dispatcher", 43, 10),
                  event("cudaLaunchKernel", 44, 1, "cuda_runtime", correlation=72),
                  gpu("opaque", 45, 20, 72), gpu("nccl_attention_name", 70, 10)]
        sample = dict(events=[api(0, None, "model_forward", 0, 100, layer=None)], cuda_device=0)
        out = K.summarize_trace(dict(traceEvents=events), sample)
        cats = out["gpu_devices"]["0"]["categories"]
        self.assertEqual(cats["attention"]["union_us"], 10)
        self.assertEqual(cats["deep_ep.dispatch"]["union_us"], 20)
        self.assertEqual(cats["unknown"]["union_us"], 10)
        self.assertTrue(out["forward_mapping"]["valid"])
        self.assertEqual(out["gpu_per_forward_layer_category"][0]["forward_index"], 1)

    def test_union_overlap_and_idle_gaps_are_separate(self):
        out = K.activity_metrics([(0, 10), (5, 15), (20, 23)])
        self.assertEqual(out["union_us"], 18)
        self.assertEqual(out["duration_sum_us"], 23)
        self.assertEqual(out["overlap_at_least_two_us"], 5)
        self.assertEqual(out["no_traced_activity_us"], 5)
        self.assertEqual(out["gaps"], [dict(start_us=15, end_us=20, duration_us=5)])
        self.assertEqual(K.intersection_length([(0, 10), (8, 12)], [(6, 15)]), 6)

    def test_distinct_devices_are_not_unioned(self):
        out = K.summarize_trace(dict(traceEvents=[gpu("x", 0, 10, device=0),
            gpu("y", 5, 10, device=1)]), dict(cuda_device=0))
        self.assertEqual(out["gpu_devices"]["0"]["activity_envelope"]["union_us"], 10)
        self.assertFalse(out["gpu_devices"]["1"]["matches_expected_device"])

    def test_kernel_copy_set_and_overlap_can_coexist(self):
        events = [gpu("kernel", 0, 10), gpu("copy", 5, 10, category="gpu_memcpy"),
                  gpu("set", 16, 2, category="gpu_memset")]
        out = K.summarize_trace(dict(traceEvents=events))
        device = out["gpu_devices"]["0"]
        self.assertEqual(device["activity_envelope"]["union_us"], 17)
        self.assertEqual(device["activity_kinds"]["kernel"]["union_us"], 10)
        self.assertEqual(device["activity_kinds"]["memcpy"]["union_us"], 10)
        self.assertEqual(device["activity_envelope"]["overlap_at_least_two_us"], 5)

    def test_ambiguous_correlation_remains_unknown(self):
        events = [event("attr::attention_full::3", 0, 10),
                  event("cudaLaunchKernel", 1, 1, "cuda_runtime", correlation=8),
                  event("attr::routed_expert_compute::runner", 20, 10),
                  event("cudaLaunchKernel", 21, 1, "cuda_runtime", correlation=8),
                  gpu("kernel", 22, 4, 8)]
        out = K.summarize_trace(dict(traceEvents=events))
        self.assertEqual(out["linkage"]["ambiguous_runtime_correlation"], 1)
        self.assertEqual(out["unknown_event_count"], 1)

    def test_external_id_is_audited_fallback(self):
        events = [event("attr::shared_expert::3", 0, 20),
                  event("aten::mm", 1, 4, "cpu_op", **{"External id": 42}),
                  gpu("x", 2, 10)]
        events[-1]["args"]["External id"] = 42
        out = K.summarize_trace(dict(traceEvents=events))
        self.assertEqual(out["linkage"]["external_id_cpu_scope"], 1)
        self.assertIn("expert.shared", out["gpu_devices"]["0"]["categories"])

    def test_cuda_event_parent_children_are_deduplicated(self):
        rows = [api(0, None, "model_forward", 0, 40, layer=None),
                api(1, 0, "layer", 0, 30, layer=None, label="3"),
                api(2, 1, "attention_full", 0, 10),
                api(3, 2, "attention_prepare", 0, 5),
                api(4, 2, "attention_core", 5, 10),
                api(5, 2, "tp_collective", 8, 10),
                api(6, 1, "routed_expert_compute", 10, 20),
                api(7, 6, "humming_gemm", 10, 17, label="w13:opaque"),
                api(8, 1, "deep_ep.combine_send", 20, 23)]
        sample = dict(events=rows, forwards=[dict(forward_index=1,
                     metadata=dict(extend_seq_lens_cpu=[16], input_ids_shape=[32]))])
        group = K.summarize_cuda_events(sample)["per_forward_layer_stream"][0]
        self.assertEqual(group["inclusive_phase_union_ms"]["attention"], 10)
        self.assertEqual(group["exclusive_api_partition_ms"]["attention"], 8)
        self.assertEqual(group["exclusive_api_partition_ms"]["tp_collective"], 2)
        self.assertEqual(group["exclusive_api_partition_ms"]["expert.routed"], 10)
        self.assertEqual(sum(group["exclusive_api_partition_ms"].values()), 30)
        self.assertEqual(group["local_real_new_tokens"], 16)
        self.assertEqual(group["per_local_real_token_ms"]["expert.routed"], 0.625)

    def test_independent_streams_zero_tokens_and_parent_cycles(self):
        rows = [api(1, None, "layer", 0, 10, layer=None, label="3"),
                api(2, 1, "attention_full", 0, 4),
                api(3, 1, "deep_ep.dispatch_recv", 1, 3)]
        rows[-1]["end_stream"] = 100
        out = K.summarize_cuda_events(dict(events=rows, forwards=[dict(forward_index=1,
                 metadata=dict(extend_seq_lens_cpu=[0]))]))
        self.assertEqual(out["excluded_cross_stream_event_indices"], [3])
        self.assertIsNone(out["per_forward_layer_stream"][0]["per_local_real_token_ms"]["attention"])
        rows[1]["parent"] = 2
        with self.assertRaises(ValueError):
            K.summarize_cuda_events(dict(events=rows))

    def test_cli_records_hashes_and_rejects_failed_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace, sample, out = root / "trace.json", root / "attribution.json", root / "out.json"
            trace.write_text(json.dumps(dict(traceEvents=[gpu("unknown", 0, 2)])))
            sample.write_text(json.dumps(dict(passed=True, events=[], rank=0, arm="synthetic")))
            self.assertEqual(K.main(["--trace",str(trace),"--attribution",str(sample),"--output",str(out)]), 0)
            result = json.loads(out.read_text())
            self.assertIsNone(result["aggregate_end_to_end_fraction"])
            self.assertEqual(result["ranks"][0]["sources"]["trace"]["bytes"], trace.stat().st_size)
            sample.write_text(json.dumps(dict(passed=False)))
            with self.assertRaises(ValueError):
                K.summarize_file(trace, sample)


if __name__ == "__main__":
    unittest.main()
