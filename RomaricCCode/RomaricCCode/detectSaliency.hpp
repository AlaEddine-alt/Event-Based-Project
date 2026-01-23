//#pragma once 

#include <cstdint>
#include <utility>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdint>


/// tarsier is a collection of event handlers.
namespace tarsier {

    /// Frame contains an image data.
    template <uint64_t width, uint64_t height>
    class Frame {
        public:
            Frame(float fill = 0) :
                data(width * height, fill)
            {
            }
            Frame(const Frame&) = default;
            Frame(Frame&&) = default;
            Frame& operator=(const Frame&) = default;
            Frame& operator=(Frame&&) = default;
            virtual ~Frame() {}
    
            /// data contains the frame values.
            std::vector<float> data;
        
            /// get retrieves a frame pixel.
            virtual float get(std::size_t x, std::size_t y) const {
                return data[x + y * width];
            }
        
            /// set retrieves a frame pixel.
            virtual void set(std::size_t x, std::size_t y, float value) {
                data[x + y * width] = value;
            }
    };
    
    /// DetectSaliency
    template <typename Event, uint64_t width, uint64_t height, unsigned int scale, uint64_t frameduration, typename HandleFrame>
    class DetectSaliency {
        public:
            DetectSaliency(HandleFrame handleFrame) :
                _handleFrame(std::forward<HandleFrame>(handleFrame)),
                _frameindex(0),
                _radius(std::pow(2, scale - 1)),
                _area(std::pow(2 * _radius + 1, 2)),
                _timewindow(10000 * std::pow(2, scale - 1)),
                _pastTimestamps(-std::numeric_limits<float>::infinity())
        
            {
            }
            DetectSaliency(const DetectSaliency&) = delete;
            DetectSaliency(DetectSaliency&&) = default;
            DetectSaliency& operator=(const DetectSaliency&) = delete;
            DetectSaliency& operator=(DetectSaliency&&) = default;
            virtual ~DetectSaliency() {}
        
            /// operator() handles an event.
            virtual void operator()(Event event) {
                std::size_t startX = (event.x >= _radius) ? event.x - _radius : 0;
                std::size_t stopX = (event.x + _radius < width ) ? event.x + _radius + 1 : width; // std::min(event.x + _radius + 1, (std::size_t)width);
                std::size_t startY = (event.y >= _radius) ? event.y - _radius : 0; // ternary operator
                std::size_t stopY = (event.y + _radius < height ) ? event.y + _radius + 1 : height; // std::min(event.y + _radius + 1, (std::size_t)height);
                uint64_t currentTimestamp = event.t;
                
                _pastTimestamps.set( event.x, event.y, (float)currentTimestamp );
                
                float saliencyValue = -1;
                int64_t areaValue = -1;
                
                for (std::size_t i = startX; i<stopX; i++) {
                    for (std::size_t j = startY; j<stopY; j++) {
                        float testValue = (float)_timewindow - (float)currentTimestamp + _pastTimestamps.get(i,j);
                        areaValue++;
                        if (  testValue > 0 ) {
                            saliencyValue++;
                        }
                    }
                }
                
                saliencyValue /= areaValue;
                
                for (std::size_t i = startX; i<stopX; i++) {
                    for (std::size_t j = startY; j<stopY; j++) {
                        _outSaliency.set(i,j,saliencyValue);
                    }
                }
                
                
                
                /// Send frame to display
                if (event.t > _frameindex * frameduration) {
                    _frameindex++;
                    _handleFrame(_outSaliency);
                }
            }
        
        protected:
            HandleFrame _handleFrame;
            uint64_t _timewindow;
            uint64_t _frameindex;
            std::size_t _radius;
            uint64_t _area;
            Frame<width,height> _pastTimestamps;
            Frame<width,height> _outSaliency;
        
            float _target;
            float _margin;
            float _alpha;
            uint64_t _sumSaliency;
            uint64_t _minSum;
            uint64_t _maxSum;
            uint64_t _targetSum;
            uint64_t _refractoryPeriod;
            uint64_t _lastUpdate;

    };
    
    /// make_DetectSaliency creates a DetectSaliency from a functor.
    template<typename Event, uint64_t width, uint64_t height, unsigned int scale, uint64_t frameduration, typename HandleFrame>
    DetectSaliency<Event, width, height, scale, frameduration, HandleFrame> make_detectSaliency(HandleFrame handleFrame) {
        return DetectSaliency<Event, width, height, scale, frameduration, HandleFrame>(std::forward<HandleFrame>(handleFrame));
    }
    
    
    /// saliencyMerger
    template < uint64_t width, uint64_t height, unsigned int nbFrames, typename HandleFrame>
    class SaliencyMerger {
    public:
        SaliencyMerger(HandleFrame handleFrame) :
            _handleFrame(std::forward<HandleFrame>(handleFrame)),
            _storeFrames(nbFrames)
        {
        }
        
        SaliencyMerger(const SaliencyMerger&) = delete;
        SaliencyMerger(SaliencyMerger&&) = default;
        SaliencyMerger& operator=(const SaliencyMerger&) = delete;
        SaliencyMerger& operator=(SaliencyMerger&&) = default;
        virtual ~SaliencyMerger() {}
        
        virtual void operator()(const Frame<width, height>& frame, unsigned int scale) {
            unsigned int frameIndx = scale - 1;
            if (frameIndx >= nbFrames) {
                throw std::logic_error("Frame index exceeds total number of frames");
            }
            
            _storeFrames[frameIndx] = frame;
            
            if (frameIndx == nbFrames - 1) {
                Frame<width,height> _outFrame;
                
                for (unsigned int k = 0; k < nbFrames; k++) {
                    for (std::size_t i = 0; i < width; i++) {
                        for (std::size_t j = 0; j < height; j++) {
                            _outFrame.set(i, j, _outFrame.get(i,j) + _storeFrames[k].get(i,j) / nbFrames);
                        }
                    }
                }
                _handleFrame(_outFrame);
            }
        }
        
    protected:
        HandleFrame _handleFrame;
        std::vector<Frame<width,height>> _storeFrames;
    };
    
    /// make_SaliencyMerger creates a SaliencyMergerfrom a functor
    template< uint64_t width, uint64_t height, unsigned int nbFrames, typename HandleFrame>
    SaliencyMerger< width, height, nbFrames, HandleFrame> make_saliencyMerger(HandleFrame handleFrame) {
        return SaliencyMerger< width, height, nbFrames, HandleFrame>(std::forward<HandleFrame>(handleFrame));
    }
}

