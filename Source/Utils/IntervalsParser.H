#ifndef WARPX_INTERVALSPARSER_H_
#define WARPX_INTERVALSPARSER_H_

#include <limits>
#include <string>
#include <vector>

/**
 * \brief This class is a parser for slices of the form i:j:k where i, j and k are integers
 * representing respectively the starting point, the stopping point and the period.
 */
class SliceParser
{
public:
    /**
    * \brief Constructor of the SliceParser class.
    *
    * @param[in] instr an input string of the form "i:j:k", "i:j" or "k" where i, j and k are
    * integers representing respectively the starting point, the stopping point and the period.
    * Any of these integers may be omitted in which case it will be equal to their default value
    * (0 for the starting point, std::numeric_limits<int>::max() for the stopping point and 1 for
    * the period). For example SliceParser(":1000:") is equivalent to SliceParser("0:1000:1").
    */
    SliceParser (const std::string& instr);

    /**
    * \brief A method that returns true if the input integer is contained in the slice. (e.g. if
    * the list is initialized with "300:500:100", this method returns true if and only if n is
    * 300, 400 or 500). If the period is negative or 0, the function always returns false.
    *
    * @param[in] n the input integer
    */
    bool contains (const int n) const;

    /**
    * \brief A method that returns the smallest integer strictly greater than n such that
    * contains(n) is true. Returns std::numeric_limits<int>::max() if there is no such integer.
    *
    * @param[in] n the input integer
    */
    int nextContains (const int n) const;

    /**
    * \brief A method that returns the greatest integer strictly smaller than n such that
    * contains(n) is true. Returns 0 if there is no such integer.
    *
    * @param[in] n the input integer
    */
    int previousContains (const int n) const;

    /**
    * \brief A method that returns the slice period.
    *
    */
    int getPeriod () const;

    /**
    * \brief A method that returns the slice start.
    *
    */
    int getStart () const;

    /**
    * \brief A method that returns the slice stop.
    *
    */
    int getStop () const;

private:
    int m_start = 0;
    int m_stop = std::numeric_limits<int>::max();
    int m_period = 1;
    std::string m_separator = ":";

};

/**
 * \brief This class is a parser for multiple slices of the form x,y,z,... where x, y and z are
 * slices of the form i:j:k, as defined in the SliceParser class. This class contains a vector of
 * SliceParsers.
 */
class IntervalsParser
{
public:
    /**
    * \brief Default constructor of the IntervalsParser class.
    */
    IntervalsParser () = default;

    /**
    * \brief Constructor of the IntervalsParser class.
    *
    * @param[in] instr_vec an input vector string, which when concatenated is of the form
    * "x,y,z,...". This will call the constructor of SliceParser using x, y and z as input
    * arguments.
    */
    IntervalsParser (const std::vector<std::string>& instr_vec);

    /**
    * \brief A method that returns true if the input integer is contained in any of the slices
    * contained by the IntervalsParser.
    *
    * @param[in] n the input integer
    */
    bool contains (const int n) const;

    /**
    * \brief A method that returns the smallest integer strictly greater than n such that
    * contains(n) is true. Returns std::numeric_limits<int>::max() if there is no such integer.
    *
    * @param[in] n the input integer
    */
    int nextContains (const int n) const;

    /**
    * \brief A method that returns the greatest integer strictly smaller than n such that
    * contains(n) is true. Returns 0 if there is no such integer.
    *
    * @param[in] n the input integer
    */
    int previousContains (const int n) const;

    /**
    * \brief A method that returns the greatest integer smaller than or equal to n such that
    * contains(n) is true. Returns 0 if there is no such integer.
    *
    * @param[in] n the input integer
    */
    int previousContainsInclusive (const int n) const;

    /**
    * \brief A method the local period (in timesteps) of the IntervalsParser at timestep n.
    * The period is defined by nextContains(n) - previousContainsInclusive(n)
    *
    * @param[in] n the input integer
    */
    int localPeriod (const int n) const;

/**
    * \brief A method that returns true if any of the slices contained by the IntervalsParser
    * has a strictly positive period.
    */
    bool isActivated () const;

private:
    std::vector<SliceParser> m_slices;
    std::string m_separator = ",";
    bool m_activated = false;
};

#endif // WARPX_INTERVALSPARSER_H_
