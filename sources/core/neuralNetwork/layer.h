#ifndef LAYER_H
#define LAYER_H

#include <assert.h>
#include <array>

#include "primitives/Matrix.h"
#include "primitives/Vector.h"

// --------------------------------------------------------------------------- 

namespace NilDa
{

enum class layerTypes {
    input, 
    dense,
    dropout, 
    flatten
};

class layer
{

protected:

    layerTypes type_;

public:

    // Constructors

    layer() {}

    // Member functions
    virtual void init(const layer* previousLayer) = 0;
   
    virtual void forwardPropagation(const Matrix& data)  = 0;

    //virtual void backwardPropagation() = 0;

    virtual int size() const = 0;

    virtual void size(std::array<int, 3>& sizes) const = 0;

    //virtual void update() = 0;    
    layerTypes layerType()  const
    {
        return type_;
    }

    // Return the string name of the type layer from the enum name
    std::string layerName(const layerTypes inLayerType) const
    {
        std::string name;

        if (inLayerType == layerTypes::input)
        {
            name = "input";
        }
        else if (inLayerType == layerTypes::dense)
        {
            name = "dense";
        }
        else if (inLayerType == layerTypes::dropout)
        {
            name = "drop out";
        }
        else if (inLayerType == layerTypes::flatten)
        {
            name = "flatten";
        }
        else
        {
            std::cerr << "Unknown layer type code." << std::endl;
            assert(false);
        }

        return name;
    }

    // Destructor
    virtual ~layer() = default;
};


} // namespace

#endif