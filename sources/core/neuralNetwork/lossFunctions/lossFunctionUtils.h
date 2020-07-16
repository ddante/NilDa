#ifndef LOSS_FUNCTION_UTILS_H
#define LOSS_FUNCTION_UTILS_H

namespace NilDa
{


enum class lossFunctions
{
    sparseCategoricalCrossentropy
};

lossFunctions lossFunctionCode(const std::string& inName)
{
    if(inName == "sparse_categorical_crossentropy")
    {
        return lossFunctions::sparseCategoricalCrossentropy;
    }
    else
    {
        std::cerr << "Unknown loss function name " 
                    << inName
                    << "." << std::endl;
        assert(false);
    }
}


} // namespace

#endif