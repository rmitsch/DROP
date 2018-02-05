export default class Utils
{
    /**
     * Generates a UUID(-like) object that can be used as quasi-unique ID for HTML elements.
     * @returns {string | * | void}
     */
    static uuidv4()
    {
        // Attach "id_" as prefix since CSS3 can't handle IDs starting with digits.
        return "id_" + ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
          (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
        )
    }

    /**
     * Returns type of object.
     * Source: https://stackoverflow.com/questions/7390426/better-way-to-get-type-of-a-javascript-variable
     * @param obj
     * @returns {string}
     */
    static toType(obj)
    {
      return ({}).toString.call(obj).match(/\s([a-zA-Z]+)/)[1].toLowerCase()
    }
}
