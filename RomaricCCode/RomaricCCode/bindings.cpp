#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <vector>

#include "detectSaliency.hpp"

namespace py = pybind11;
using namespace tarsier;

/* ---------------- CONFIG ---------------- */
constexpr uint64_t WIDTH  = 128;
constexpr uint64_t HEIGHT = 128;
constexpr unsigned int SCALE = 3;
constexpr uint64_t FRAME_DURATION = 1000;
constexpr unsigned int NB_FRAMES = 3;
/* ---------------------------------------- */

struct Event {
    uint16_t x;
    uint16_t y;
    uint64_t t;
};

/* ---------- Frame wrapper ---------- */
using CppFrame = Frame<WIDTH, HEIGHT>;

py::array_t<float> frame_to_numpy(const CppFrame& f) {
    return py::array_t<float>(
        {HEIGHT, WIDTH},
        f.data.data()
    );
}

/* ---------- DetectSaliency wrapper ---------- */
class PyDetectSaliency {
public:
    PyDetectSaliency(py::function callback)
        : _callback(callback),
          _detector([this](const CppFrame& frame) {
              py::gil_scoped_acquire gil;
              _callback(frame_to_numpy(frame));
          })
    {}

    void feed(uint16_t x, uint16_t y, uint64_t t) {
        _detector(Event{x, y, t});
    }

private:
    py::function _callback;

    DetectSaliency<
        Event,
        WIDTH,
        HEIGHT,
        SCALE,
        FRAME_DURATION,
        std::function<void(const CppFrame&)>
    > _detector;
};

/* ---------- SaliencyMerger wrapper ---------- */
class PySaliencyMerger {
public:
    PySaliencyMerger(py::function callback)
        : _callback(callback),
          _merger([this](const CppFrame& frame) {
              py::gil_scoped_acquire gil;
              _callback(frame_to_numpy(frame));
          })
    {}

    void feed(py::array_t<float> frame, unsigned int scale) {
        auto buf = frame.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("Frame must be 2D");
        }

        CppFrame cppFrame;
        std::memcpy(cppFrame.data.data(), buf.ptr,
                    sizeof(float) * WIDTH * HEIGHT);

        _merger(cppFrame, scale);
    }

private:
    py::function _callback;

    SaliencyMerger<
        WIDTH,
        HEIGHT,
        NB_FRAMES,
        std::function<void(const CppFrame&)>
    > _merger;
};

/* ---------- Module ---------- */
PYBIND11_MODULE(saliency, m) {
    py::class_<Event>(m, "Event")
        .def(py::init<>())
        .def_readwrite("x", &Event::x)
        .def_readwrite("y", &Event::y)
        .def_readwrite("t", &Event::t);

    py::class_<PyDetectSaliency>(m, "DetectSaliency")
        .def(py::init<py::function>())
        .def("feed", &PyDetectSaliency::feed);

    py::class_<PySaliencyMerger>(m, "SaliencyMerger")
        .def(py::init<py::function>())
        .def("feed", &PySaliencyMerger::feed);
}