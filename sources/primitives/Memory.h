#ifndef MEMORY_H
#define MEMORY_H

template<class outT, class inT>
outT memcpy_reverse(outT dest, inT first, inT last)
{
    while(last != first)
    {
        *dest = *(--last);
        ++dest;
    }

    return dest;
}

#endif
