#ifndef LAYER_H
#define LAYER_H

#include <assert.h>
#include <array>

// --------------------------------------------------------------------------- 

namespace NilDa
{

enum LayerTypes {
    LAYER_TYPE_INPUT, 
    LAYER_TYPE_DENSE, 
    LAYER_TYPE_DROPOUT, 
    LAYER_TYPE_FLATTEN
};

class layer
{

protected:

    LayerTypes type_;

public:

    // Constructors

    layer() {}

    // Member functions
    virtual void init(const layer* previousLayer) = 0;
   
    virtual int size() const = 0;

    virtual void size(std::array<int, 3>& sizes) const = 0;

    //virtual void forwardPropagation()  = 0;

    //virtual void backwardPropagation() = 0;

    //virtual void update() = 0;    
    LayerTypes layerType()  const
    {
        return type_;
    }

    std::string layerName(const LayerTypes inLayerType) const
    {
        std::string name;

        if (inLayerType == LAYER_TYPE_INPUT)
        {
            name = "input";
        }
        else if (inLayerType == LAYER_TYPE_DENSE)
        {
            name = "dense";
        }
        else if (inLayerType == LAYER_TYPE_DROPOUT)
        {
            name = "drop out";
        }
        else if (inLayerType == LAYER_TYPE_FLATTEN)
        {
            name = "flatten";
        }
        else
        {
            std::cerr << "Uknown layer type code " 
                        << inLayerType 
                        << "." << std::endl;
            assert(false);
        }

        return name;
    }

    // Destructor
    virtual ~layer() {}
};


} // namespace

#endif