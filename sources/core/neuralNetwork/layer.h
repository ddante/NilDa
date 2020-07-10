#ifndef LAYER_H
#define LAYER_H

// --------------------------------------------------------------------------- 

namespace NilDa
{

enum LayerTypes {
    INPUT, 
    DENSE, 
    DROPOUT, 
    FLATTEN
};

class layer
{
public:

    LayerTypes layerType;

public:

    // Constructors

    layer() {}

    // Member functions

    virtual void init(const layer* previousLayer) = 0;
   
    virtual int size() const = 0;

    //virtual void forwardPropagation()  = 0;

    //virtual void backwardPropagation() = 0;

    //virtual void update() = 0;    

    // Destructor
    virtual ~layer() {}

};


}

#endif