import warnings

import numpy as np
import pandas as pd
import scipy.ndimage
import skimage

import matplotlib._contour
from matplotlib.pyplot import get_cmap as mpl_get_cmap

import bokeh.models
import bokeh.palettes
import bokeh.plotting

import altair as alt

def _outliers(data):
    bottom, middle, top = np.percentile(data, [25, 50, 75])
    iqr = top - bottom
    top_whisker = min(top + 1.5*iqr, data.max())
    bottom_whisker = max(bottom - 1.5*iqr, data.min())
    outliers = data[(data > top_whisker) | (data < bottom_whisker)]
    return outliers

def _box_and_whisker(data):
    middle = data.median()
    bottom = data.quantile(0.25)    
    top = data.quantile(0.75)
    iqr = top - bottom
    top_whisker = min(top + 1.5*iqr, data.max())
    bottom_whisker = max(bottom - 1.5*iqr, data.min())
    return pd.Series({'middle': middle, 
                      'bottom': bottom, 
                      'top': top, 
                      'top_whisker': top_whisker, 
                      'bottom_whisker': bottom_whisker})

def _jitter(x, jitter_width=0.2):
    """Make x-coordinates for a jitter plot."""
    return (pd.Categorical(x).codes 
            + np.random.uniform(low=-jitter_width,
                                high=jitter_width,
                                size=len(x)))


def _convert_data(data, inf_ok=False, min_len=1):
    """
    Convert inputted 1D data set into NumPy array of floats.
    All nan's are dropped.

    Parameters
    ----------
    data : int, float, or array_like
        Input data, to be converted.
    inf_ok : bool, default False
        If True, np.inf values are allowed in the arrays.
    min_len : int, default 1
        Minimum length of array.

    Returns
    -------
    output : ndarray
        `data` as a one-dimensional NumPy array, dtype float.
    """
    # If it's scalar, convert to array
    if np.isscalar(data):
        data = np.array([data], dtype=np.float)

    # Convert data to NumPy array
    data = np.array(data, dtype=np.float)

    # Make sure it is 1D
    if len(data.shape) != 1:
        raise RuntimeError('Input must be a 1D array or Pandas series.')

    # Remove NaNs
    data = data[~np.isnan(data)]

    # Check for infinite entries
    if not inf_ok and np.isinf(data).any():
        raise RuntimeError('All entries must be finite.')

    # Check to minimal length
    if len(data) < min_len:
        raise RuntimeError('Array must have at least {0:d} non-NaN entries.'.format(min_len))

    return data


def ecdf_vals(data, formal=False, x_min=None, x_max=None):
    """Get x, y, values of an ECDF for plotting.

    Parameters
    ----------
    data : ndarray
        One dimensional Numpay array with data.
    formal : bool, default False
        If True, generate x and y values for formal ECDF (staircase). If
        False, generate x and y values for ECDF as dots.
    x_min : float, 'infer', or None
        Minimum value of x to plot. If 'infer', use a 5% buffer. Ignored
        if `formal` is False.
    x_max : float, 'infer', or None
        Maximum value of x to plot. If 'infer', use a 5% buffer. Ignored
        if `formal` is False.

    Returns
    -------
    x : ndarray
        x-values for plot
    y : ndarray
        y-values for plot
    """
    x = np.sort(data)
    y = np.arange(1, len(data)+1) / len(data)

    if formal:
        # Set up output arrays
        x_formal = np.empty(2*(len(x) + 1))
        y_formal = np.empty(2*(len(x) + 1))

        # y-values for steps
        y_formal[:2] = 0
        y_formal[2::2] = y
        y_formal[3::2] = y

        # x- values for steps
        x_formal[0] = x[0]
        x_formal[1] = x[0]
        x_formal[2::2] = x
        x_formal[3:-1:2] = x[1:]
        x_formal[-1] = x[-1]
        
        # Put lines at y=0
        if x_min is not None:
            if x_min == 'infer':
                x_min = x.min() - (x.max() - x.min())*0.05
            elif x_min > x.min():
                raise RuntimeError('x_min > x.min().')
            x_formal = np.concatenate(((x_min,), x_formal))
            y_formal = np.concatenate(((0,), y_formal))

        # Put lines at y=y.max()
        if x_max is not None:
            if x_max == 'infer':
                x_max = x.max() + (x.max() - x.min())*0.05
            elif x_max < x.max():
                raise RuntimeError('x_max < x.max().')
            x_formal = np.concatenate((x_formal, (x_max,)))
            y_formal = np.concatenate((y_formal, (y.max(),)))

        return x_formal, y_formal
    else:
        return x, y


def ecdf_y(data):
    """Give y-values of an ECDF for an unsorted column in a data frame.
    
    Parameters
    ----------
    data : Pandas Series
        Series (or column of a DataFrame) from which to generate ECDF
        values

    Returns
    -------
    output : Pandas Series
        Corresponding y-values for an ECDF when plotted with dots.

    Notes
    -----
    .. This only works for plotting an ECDF with points, not for formal
       ECDFs
    """
    return data.rank(method='first') / len(data)


def ecdf_dataframe(data=None, x=None, color=None, formal=False):
    """Generate a DataFrame that can be used for plotting ECDFs.

    Parameters
    ----------
    data : Pandas DataFrame
        A tidy data frame.
    x : valid column name of Pandas DataFrame
        Column of data frame containing values to use in ECDF plot.
    color : valid column name of Pandas DataFrame or list of column 
            names
        Column(s) of DataFrame to use for grouping the data. A unique
        set of ECDF values is made for each. If None, no groupby 
        operations are performed and a single ECDF is generated.
    formal : bool, default False
        If True, generate x and y values for formal ECDF (staircase). If
        False, generate x and y values for ECDF as dots.


    Returns
    -------
    output : Pandas DataFrame
        Pandas DataFrame with two or three columns.
            x : Column named for inputted `x`, data values.
            'ECDF': Values for y-values for plotting the ECDF
            color : Keys for groups. Omitted if `color` is None.
    """
    if data is None:
        raise RuntimeError('`data` must be specified.')
    if x is None:
        raise RuntimeError('`x` must be specified.')

    # Determine ranges of plots
    if formal:
        data_min = data[x].min()
        data_max = data[x].max()
        x_min = data_min - (data_max - data_min) * 0.05
        x_max = data_max + (data_max - data_min) * 0.05
    else:
        x_min = None
        x_max = None

    if color is None:
        x_ecdf, y_ecdf = ecdf_vals(data[x].values,
                                   formal=formal, 
                                   x_min=x_min, 
                                   x_max=x_max)
        return pd.DataFrame({x: x_ecdf, 'ECDF': y_ecdf})
    else:
        grouped = data.groupby(color)
        df_list = []
        for g in grouped:
            if type(g[0]) == tuple:
                cat = ', '.join([str(c) for c in g[0]])
            else:
                cat = g[0]

            x_ecdf, y_ecdf = ecdf_vals(g[1][x],
                                       formal=formal,
                                       x_min=x_min, 
                                       x_max=x_max)

            df_list.append(pd.DataFrame(data={color: [cat]*len(x_ecdf),
                                              x: x_ecdf, 
                                              'ECDF': y_ecdf}))

        return pd.concat(df_list, ignore_index=True)


def altair_jitter(data=None, encode_x=None, encode_y=None, 
                  encode_tooltip=alt.Tooltip(),
                  height=alt.utils.schemapi.Undefined,
                  width=alt.utils.schemapi.Undefined, jitter_width=0.2):
    """Generate a jitter plot with Altair.

    Parameters
    ----------
    data : Pandas DataFrame
        A tidy data frame.
    encode_x : str or altair.X instance
        Vega-Lite specification of x-values.
    encode_y : str or altair.Y instance
        Vega-Lite specification of y-values.
    encode_tooltip : list or altair.Tooltip instance
        Specification for tooltips.
    height : float or Undefined (default)
        Height of the chart, in pixels.
    width : float or Undefined (default)
        Width of the chart, in pixels.
    jitter_width : float
        Maximum jitter distance; must be between 0 and 0.5 to avoid 
        clashes.

    Returns
    -------
    output : Chart
        Altair Chart instance.
    """
    if data is None:
        raise RuntimeError('`data` must be specified.')
    if encode_x is None:
        raise RuntimeError('`encode_x` must be specified.')
    if encode_y is None:
        raise RuntimeError('`encode_y` must be specified.')
    if not (0 <= jitter_width <= 0.5):
        raise RuntimeError('Must have `jitter_width` between 0 and 0.5.')

    # Make Altair instances
    if isinstance(encode_x, alt.X):
        x = encode_x
    else:
        x = alt.X(encode_x)

    if isinstance(encode_y, alt.Y):
        y = encode_y
    else:
        y = alt.Y(encode_y)

    # Get column names
    if len(x.shorthand) > 1 and x.shorthand[-2] == ':':
        x_name = x.shorthand[:-2]
    else:
        x_name = x.shorthand

    if len(y.shorthand) > 1 and y.shorthand[-2] == ':':
        y_name = y.shorthand[:-2]
    else:
        y_name = y.shorthand

    # Determine types
    var_types = [None, None]
    for i, var in enumerate([x, y]):
        if not isinstance(var.type, alt.utils.schemapi.UndefinedType):
            var_types[i] = var.type[0].upper()
        elif len(var.shorthand) > 1 and var.shorthand[-2] == ':':
            var_types[i] = var.shorthand[-1]
        else:
            raise RuntimeError(
                    f'Data type of `encode_{var}` must be specified.')

    # Make sure data types are given and ok
    if var_types[0] not in 'NO' and var_types[1] not in 'NO':
        raise RuntimeError('Either `x` or `y` must be nominal or ordinal.')
    if var_types == ['N, N']:
        raise RuntimeError('Cannot have both `x` and `y` be nominal.')

    # Decide if it's a horizontal plot or not
    if var_types[0] in 'NO':
        horizontal = False
        cats = x_name
        val = y_name
        if isinstance(y.title, alt.utils.schemapi.UndefinedType):
            y.title = y_name
    else:
        horizontal = True
        cats = y_name
        val = x_name
        if isinstance(x.title, alt.utils.schemapi.UndefinedType):
            x.title = x_name

    # Copy DataFrame so we don't overwrite anything
    df = data.copy()

    # Set up groupby object
    n_cats = len(df[cats].unique())
    nominal_axis_vals = list(range(n_cats))

    # Make coordinates for plotting
    df['__jitter'] = _jitter(df[cats], jitter_width)


    if horizontal:
        chart = alt.Chart(
                data=df,
                width=width,
                height=height              
            ).mark_point(
            ).encode(
                y=alt.Y(
                    '__jitter:Q',
                    axis=alt.Axis(
                        title=None,
                        values=nominal_axis_vals,
                        labels=False,
                        grid=False,
                        ticks=False)),
                x=x,
                color=alt.Color(f'{cats}:N', title=y.title),
                tooltip=encode_tooltip)
    else:
        chart = alt.Chart(
                data=df,
                width=width,
                height=height              
            ).mark_point(
            ).encode(
               x=alt.X(
                    '__jitter:Q',
                    axis=alt.Axis(
                        title=None,
                        values=nominal_axis_vals,
                        labels=False,
                        grid=False,
                        ticks=False)),
                y=y,
                color=alt.Color(f'{cats}:N', title=x.title),
                tooltip=encode_tooltip)
    return chart


def altair_box(data=None, encode_x=None, encode_y=None, 
               encode_color=alt.Color(), height=None, width=None):
    """Generate a box plot with Altair.

    Parameters
    ----------
    data : Pandas DataFrame
        A tidy data frame.
    encode_x : str or altair.X instance
        Specification of x-values.
    encode_y : str or altair.Y instance
        Specification of y-values.
    encode_color : str or Color instance or None or Undefined (default)
        Specification of coloring of box plot. If Undefined (Default),
        all boxes are colored with Altair defaults. If None, the boxes
        are colored according to the categorical variable.
    height : float or None (default)
        Height of the chart, in pixels. If None, inferred.
    width : float or None (default)
        Width of the chart, in pixels. If None, inferred.

    Returns
    -------
    output : Chart
        Altair Chart instance.
    """

    # Make Altair instances
    if isinstance(encode_x, alt.X):
        x = encode_x
    else:
        x = alt.X(encode_x)

    if isinstance(encode_y, alt.Y):
        y = encode_y
    else:
        y = alt.Y(encode_y)

    # Get column names
    if len(x.shorthand) > 1 and x.shorthand[-2] == ':':
        x_name = x.shorthand[:-2]
    else:
        x_name = x.shorthand

    if len(y.shorthand) > 1 and y.shorthand[-2] == ':':
        y_name = y.shorthand[:-2]
    else:
        y_name = y.shorthand

    # Get axis titles
    if isinstance(x.title, alt.utils.schemapi.UndefinedType):
        x_title = x_name
    else:
        x_title = x.title
    if isinstance(y.title, alt.utils.schemapi.UndefinedType):
        y_title = y_name
    else:
        y_title = y.title

    # Determine types
    var_types = [None, None]
    for i, var in enumerate([x, y]):
        if not isinstance(var.type, alt.utils.schemapi.UndefinedType):
            var_types[i] = var.type[0].upper()
        elif len(var.shorthand) > 1 and var.shorthand[-2] == ':':
            var_types[i] = var.shorthand[-1]
        else:
            raise RuntimeError(
                    f'Data type of `encode_{var}` must be specified.')

    # Make sure data types are given and ok
    if var_types[0] not in 'NO' and var_types[1] not in 'NO':
        raise RuntimeError('Either `x` or `y` must be nominal or ordinal.')
    if var_types == ['N, N']:
        raise RuntimeError('Cannot have both `x` and `y` be nominal.')

    # Decide if it's a horizontal plot or not
    if var_types[0] in 'NO':
        horizontal = False
        cats = x_name
        val = y_name
        if encode_color is None:
            encode_color = alt.Color(f'{cats}:N', title=x.title)
    else:
        horizontal = True
        cats = y_name
        val = x_name
        if encode_color is None:
            encode_color = alt.Color(f'{cats}:N', title=y.title)

    # Set up groupby object
    grouped = data.groupby(cats)
    n_boxes = len(grouped)

    # Set default heights and widths, also of bars
    if width is None:
        if horizontal:
            width = 400
        else:
            width = 200
    if height is None:
        if horizontal:
            height = 200
        else:
            height = 300

    if horizontal:
        size = height*0.9 / n_boxes
    else:
        size = width*0.9 / n_boxes
            
    # Data frame for boxes and whiskers
    df_box = (grouped[val].apply(_box_and_whisker)
                          .reset_index()
                          .rename(columns={'level_1': 'box_val'})
                          .pivot(index=cats, columns='box_val'))
    df_box.columns = df_box.columns.get_level_values(1)
    df_box = df_box.reset_index()

    # Data frame for outliers
    df_outlier = grouped[val].apply(_outliers).reset_index(level=0)

    if horizontal:
        chart_box = alt.Chart(
                data=df_box,
                width=width,
                height=height              
            ).mark_bar(
                size=size
            ).encode(
                y=alt.Y(f'{cats}:N', title=y_title),
                x=alt.X('bottom:Q', title=x_title),
                x2=alt.X2('top:Q', title=x_title),
                color=encode_color)

        chart_median = alt.Chart(
                data=df_box,
                width=width,
                height=height              
            ).mark_tick(
                size=size,
                color='white'
            ).encode(
                y=alt.Y(f'{cats}:N',  title=y_title),
                x=alt.X('middle:Q', title=x_title))

        chart_whisker = alt.Chart(
                data=df_box,
                width=width,
                height=height              
            ).mark_rule(
            ).encode(
                y=alt.Y(f'{cats}:N', title=y_title),
                x=alt.X('bottom_whisker:Q', title=x_title),
                x2=alt.X2('top_whisker:Q', title=x_title))

        chart_outliers = alt.Chart(
                data=df_outlier,
                width=width,
                height=height              
            ).mark_point(
            ).encode(
                y=alt.Y(f'{cats}:N', title=y_title),
                x=alt.X(f'{val}:Q', title=x_title),
                color=encode_color)
    else:
        chart_box = alt.Chart(
                data=df_box,
                width=width,
                height=height       
            ).mark_bar(
                size=size
            ).encode(
                x=alt.X(f'{cats}:N', title=x_title),
                y=alt.Y('bottom:Q', title=y_title),
                y2=alt.Y2('top:Q', title=y_title),
                color=encode_color)

        chart_median = alt.Chart(
                data=df_box,
                width=width,
                height=height       
            ).mark_tick(
                size=size,
                color='white'
            ).encode(
                x=alt.X(f'{cats}:N', title=x_title),
                y=alt.Y('middle:Q', title=y_title))

        chart_whisker = alt.Chart(
                data=df_box,
                width=width,
                height=height       
            ).mark_rule(
            ).encode(
                x=alt.X(f'{cats}:N', title=x_title),
                y=alt.Y('bottom_whisker:Q', title=y_title),
                y2=alt.Y2('top_whisker:Q', title=y_title))

        chart_outliers = alt.Chart(
                data=df_outlier,
                width=width,
                height=height       
            ).mark_point(
            ).encode(
                x=alt.X(f'{cats}:N', title=x_title),
                y=alt.Y(f'{val}:Q', title=y_title),
                color=encode_color)

    return chart_whisker + chart_box + chart_median + chart_outliers


def bokeh_fill_between(x1, y1, x2, y2, x_axis_label=None, y_axis_label=None,
                       x_axis_type='linear', y_axis_type='linear',
                       title=None, plot_height=300, plot_width=450,
                       fill_color='#1f77b4', line_color='#1f77b4',
                       show_line=True, line_width=1, fill_alpha=1,
                       line_alpha=1, p=None, **kwargs):
    """
    Create a filled region between two curves.

    Parameters
    ----------
    x1 : array_like
        Array of x-values for first curve
    y1 : array_like
        Array of y-values for first curve
    x2 : array_like
        Array of x-values for second curve
    y2 : array_like
        Array of y-values for second curve
    x_axis_label : str, default None
        Label for the x-axis. Ignored is `p` is not None.
    y_axis_label : str, default None
        Label for the y-axis. Ignored is `p` is not None.
    x_axis_type : str, default 'linear'
        Either 'linear' or 'log'.
    y_axis_type : str, default 'linear'
        Either 'linear' or 'log'.    title : str, default None
        Title of the plot. Ignored is `p` is not None.
    plot_height : int, default 300
        Height of plot, in pixels. Ignored is `p` is not None.
    plot_width : int, default 450
        Width of plot, in pixels. Ignored is `p` is not None.
    fill_color : str, default '#1f77b4'
        Color of fill as a hex string.
    line_color : str, default '#1f77b4'
        Color of the line as a hex string.
    show_line : bool, default True
        If True, show the lines on the edges of the fill.
    line_width : int, default 1
        Line width of lines on the edgs of the fill.
    fill_alpha : float, default 1.0
        Opacity of the fill.
    line_alpha : float, default 1.0
        Opacity of the lines.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with fill-between.

    Notes
    -----
    .. Any remaining kwargs are passed to bokeh.models.patch().
    """
    if p is None:
        p = bokeh.plotting.figure(
            plot_height=plot_height, plot_width=plot_width,
            x_axis_type=x_axis_type, y_axis_type=y_axis_type,
            x_axis_label=x_axis_label, y_axis_label=y_axis_label, title=title)


    p.patch(x=np.concatenate((x1, x2[::-1])),
            y=np.concatenate((y1, y2[::-1])),
            alpha=fill_alpha,
            fill_color=fill_color,
            line_width=0,
            **kwargs)

    if show_line:
        p.line(x1,
               y1, 
               line_width=line_width, 
               alpha=line_alpha, 
               color=line_color)
        p.line(x2, 
               y2, 
               line_width=line_width, 
               alpha=line_alpha, 
               color=line_color)

    return p


def bokeh_ecdf(data, p=None, x_axis_label=None, y_axis_label='ECDF',
               title=None, plot_height=300, plot_width=450, formal=False, 
               x_axis_type='linear', y_axis_type='linear', **kwargs):
    """
    Create a plot of an ECDF.

    Parameters
    ----------
    data : array_like
        One-dimensional array of data. Nan's are ignored.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    x_axis_label : str, default None
        Label for the x-axis. Ignored is `p` is not None.
    y_axis_label : str, default 'ECDF'
        Label for the y-axis. Ignored is `p` is not None.
    title : str, default None
        Title of the plot. Ignored is `p` is not None.
    plot_height : int, default 300
        Height of plot, in pixels. Ignored is `p` is not None.
    plot_width : int, default 450
        Width of plot, in pixels. Ignored is `p` is not None.
    formal : bool, default False
        If True, make a plot of a formal ECDF (staircase). If False,
        plot the ECDF as dots.
    x_axis_type : str, default 'linear'
        Either 'linear' or 'log'.
    y_axis_type : str, default 'linear'
        Either 'linear' or 'log'.
    kwargs
        Any kwargs to be passed to either p.circle or p.line, for
        `formal` being False or True, respectively.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with ECDF.
    """
    # Check data to make sure legit
    data = _convert_data(data)

    # Data points on ECDF
    x, y = ecdf_vals(data, formal)

    # Instantiate Bokeh plot if not already passed in
    if p is None:
        p = bokeh.plotting.figure(
            plot_height=plot_height, plot_width=plot_width, 
            x_axis_label=x_axis_label, y_axis_label=y_axis_label,
            x_axis_type=x_axis_type, y_axis_type=y_axis_type, title=title)

    if formal:
        # Line of steps
        p.line(x, y, **kwargs)

        # Rays for ends
        p.ray(x[0], 0, None, np.pi, **kwargs)
        p.ray(x[-1], 1, None, 0, **kwargs)      
    else:
        p.circle(x, y, **kwargs)

    return p


def bokeh_histogram(data, bins=10, p=None, x_axis_label=None,
                    y_axis_label=None, title=None, plot_height=300,
                    plot_width=450, density=True, kind='step', **kwargs):
    """
    Make a plot of a histogram of a data set.

    Parameters
    ----------
    data : array_like
        1D array of data to make a histogram out of
    bins : int or array_like, default 10
        Setting for `bins` kwarg to be passed to `np.histogram()`.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    x_axis_label : str, default None
        Label for the x-axis. Ignored is `p` is not None.
    y_axis_label : str, default None
        Label for the y-axis. Ignored is `p` is not None.
    title : str, default None
        Title of the plot. Ignored is `p` is not None.
    plot_height : int, default 300
        Height of plot, in pixels. Ignored is `p` is not None.
    plot_width : int, default 450
        Width of plot, in pixels. Ignored is `p` is not None.
    density : bool, default True
        If True, normalized the histogram. Otherwise, base the histogram
        on counts.
    kind : str, default 'step'
        The kind of histogram to display. Allowed values are 'step' and
        'step_filled'.

    Returns
    -------
    output : Bokeh figure
        Figure populted with histogram.
    """
   # Instantiate Bokeh plot if not already passed in
    if p is None:
        p = bokeh.plotting.figure(
            plot_height=plot_height, plot_width=plot_width, 
            x_axis_label=x_axis_label, y_axis_label=y_axis_label, title=title)

    # Compute histogram
    f, e = np.histogram(data, bins=bins, density=density)
    e0 = np.empty(2*len(e))
    f0 = np.empty(2*len(e))
    e0[::2] = e
    e0[1::2] = e
    f0[0] = 0
    f0[-1] = 0
    f0[1:-1:2] = f
    f0[2:-1:2] = f

    if kind == 'step':
        p.line(e0, f0, **kwargs)

    if kind == 'step_filled':
        x2 = [e0.min(), e0.max()]
        y2 = [0, 0]
        p = fill_between(e0, f0, x2, y2, show_line=True, p=p, **kwargs)

    return p


def _catplot(df, cats, val, kind, p=None, x_axis_label=None,
             y_axis_label=None, title=None, plot_height=300, plot_width=400, 
             palette=bokeh.palettes.d3['Category10'][10],
             show_legend=False, formal=False, width=0.5, order=None,
             x_axis_type='linear', y_axis_type='linear', **kwargs):
    """
    Generate a plot with a categorical variable on x-axis.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame containing tidy data for plotting.
    cats : hashable or list of hastables
        Name of column(s) to use as categorical variable (x-axis). This is
        akin to a kdim in HoloViews.
    val : hashable
        Name of column to use as value variable. This is akin to a vdim
        in HoloViews.
    kind : str, either 'jitter' or 'box'
        Kind of plot to make.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    x_axis_label : str, default None
        Label for the x-axis. Ignored is `p` is not None.
    y_axis_label : str, default 'ECDF'
        Label for the y-axis. Ignored is `p` is not None.
    title : str, default None
        Title of the plot. Ignored is `p` is not None.
    plot_height : int, default 300
        Height of plot, in pixels. Ignored is `p` is not None.
    plot_width : int, default 450
        Width of plot, in pixels. Ignored is `p` is not None.
    palette : list of strings of hex colors, or since hex string
        If a list, color palette to use. If a single string representing
        a hex color, all glyphs are colored with that color. Default is
        the default color cycle employed by HoloViews.
    show_legend : bool, default False
        If True, show legend.
    width : float, default 0.5
        Maximum allowable width of jittered points or boxes. A value of
        1 means that the points or box take the entire space allotted.
    formal : bool, default False
        If True, make a plot of a formal ECDF (staircase). If False,
        plot the ECDF as dots. Only active when `kind` is 'ecdf'.
    show_legend : bool, default False
        If True, show a legend. Only active when `kind` is 'ecdf' or
        'colored_ecdf'.
    order : list or None
        If not None, must be a list of unique entries in `df[val]`. The
        order of the list specifies the order of the boxes. If None,
        the boxes appear in the order in which they appeared in the
        inputted DataFrame.
    x_axis_type : 'linear' or 'log'
        Type of x-axis.
    y_axis_type : 'linear' or 'log'
        Type of y-axis.
    kwargs
        Any kwargs to be passed to p.circle when making the jitter plot
        or to p.quad when making a box plot..

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with jitter plot or box plot.
    """
    if order is not None:
        if len(order) > len(set(order)):
            raise RuntimeError('Nonunique entries in `order`.')

    if formal == True and kind != 'ecdf':
        warnings.warn('`formal` kwarg not active for ' + kind + '.')
    if show_legend == True and kind not in ['ecdf', 'colored_ecdf']:
        warnings.warn('`show_legend` kwarg not active for ' + kind + '.')

    if p is None:
        if y_axis_label is None and kind not in ['ecdf', 'colored_ecdf']:
            y_axis_label = val
            
        p = bokeh.plotting.figure(
            plot_height=plot_height, plot_width=plot_width, 
            x_axis_label=x_axis_label, y_axis_label=y_axis_label,
            x_axis_type=x_axis_type, y_axis_type=y_axis_type, title=title)

        p_was_None = True
    else:
        p_was_None = False

    # Get GroupBy object, sorted if need be
    if kind == 'colored_ecdf':
        df_sorted = df.sort_values(by=val)
        _, df_sorted['__ecdf_y_values'] = ecdf_vals(df_sorted[val])
        gb = df_sorted.groupby(cats)
    else:
        gb = df.groupby(cats)

    # Number of categorical variables
    n = len(gb)
        
    # If a single string for palette, set color
    if type(palette) == str:
        if kind  != 'box' and 'color' not in kwargs:
            kwargs['color'] = palette
        elif kind == 'box' and 'fill_color' not in kwargs:
            kwargs['fill_color'] = palette
        palette = None
    elif len(palette) == 1:
        if kind != 'box' and 'color' not in kwargs:
            kwargs['color'] = palette[0]
        elif kind == 'box' and 'fill_color' not in kwargs:
            kwargs['fill_color'] = palette[0]
        palette = None
    else:
        color_cycle = list(range(len(palette))) * (n // len(palette) + 1)

    # Set box line colors
    if kind == 'box' and 'line_color' not in kwargs:
        kwargs['line_color'] = 'black'

    # Set up the iterator over the groupby object
    if order is None:
        order = list(gb.groups.keys())
    gb_iterator = [(order_val, gb.get_group(order_val)) 
                        for order_val in order]

    labels = {}
    for i, g in enumerate(gb_iterator):
        if kind in ['box', 'jitter']:
            x = i + 0.5

            if type(g[0]) == tuple:
                labels[x] = ', '.join([str(c) for c in g[0]])
            else:
                labels[x] = str(g[0])

        if kind == 'box':
            data = g[1][val]
            bottom, middle, top = np.percentile(data, [25, 50, 75])
            iqr = top - bottom
            left = x - width / 2
            right = x + width / 2
            top_whisker = min(top + 1.5*iqr, data.max())
            bottom_whisker = max(bottom - 1.5*iqr, data.min())
            whisk_lr = [x - 0.1, x + 0.1]
            outliers = data[(data > top_whisker) | (data < bottom_whisker)]

            if palette is None:
                p.quad(left, right, top, bottom, **kwargs)
            else:
                p.quad(left, right, top, bottom,
                       fill_color=palette[color_cycle[i]], **kwargs)
            p.line([left, right], [middle]*2, color='black')
            p.line([x, x], [bottom, bottom_whisker], color='black')
            p.line([x, x], [top, top_whisker], color='black')
            p.line(whisk_lr, bottom_whisker, color='black')
            p.line(whisk_lr, top_whisker, color='black')
            p.circle([x]*len(outliers), outliers, color='black')
        elif kind == 'jitter':
            if palette is None:
                p.circle(x={'value': x, 
                            'transform': bokeh.models.Jitter(width=width)},
                         y=g[1][val],
                         **kwargs)
            else:
                p.circle(x={'value': x, 
                            'transform': bokeh.models.Jitter(width=width)},
                         y=g[1][val], 
                         color=palette[color_cycle[i]],
                         **kwargs)
        elif kind in ['ecdf', 'colored_ecdf']:
            if show_legend:
                if type(g[0]) == tuple:
                    legend = ', '.join([str(c) for c in g[0]])
                else:
                    legend = str(g[0])
            else:
                legend = None

            if kind == 'ecdf':
                if palette is None:
                    bokeh_ecdf(g[1][val],
                         formal=formal,
                         p=p, 
                         legend=legend, 
                         **kwargs)
                else:
                    bokeh_ecdf(g[1][val],
                         formal=formal,
                         p=p,
                         legend=legend,
                         color=palette[color_cycle[i]],
                         **kwargs)
            elif kind == 'colored_ecdf':
                if palette is None:
                    p.circle(g[1][val],
                             g[1]['__ecdf_y_values'],
                             legend=legend, 
                             **kwargs)
                else:
                    p.circle(g[1][val],
                             g[1]['__ecdf_y_values'],
                             legend=legend, 
                             color=palette[color_cycle[i]],
                             **kwargs)
   
    if kind in ['box', 'jitter']:
        p.xaxis.ticker = np.arange(len(gb)) + 0.5
        p.xaxis.major_label_overrides = labels
        p.xgrid.visible = False
        
    if kind in ['ecdf', 'colored_ecdf']:
        p.legend.location = 'bottom_right'

    return p


def bokeh_ecdf_collection(
        df, cats, val, p=None, x_axis_label=None, y_axis_label=None,
        title=None, plot_height=300, plot_width=400, 
        palette=bokeh.palettes.d3['Category10'][10],
        show_legend=True, formal=False, order=None, x_axis_type='linear',
        **kwargs):
    """
    Make a collection of ECDFs from a tidy DataFrame.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame containing tidy data for plotting.
    cats : hashable or list of hastables
        Name of column(s) to use as categorical variable (x-axis). This is
        akin to a kdim in HoloViews.
    val : hashable
        Name of column to use as value variable. This is akin to a vdim
        in HoloViews.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    x_axis_label : str, default None
        Label for the x-axis. Ignored is `p` is not None.
    y_axis_label : str, default 'ECDF'
        Label for the y-axis. Ignored is `p` is not None.
    title : str, default None
        Title of the plot. Ignored is `p` is not None.
    plot_height : int, default 300
        Height of plot, in pixels. Ignored is `p` is not None.
    plot_width : int, default 450
        Width of plot, in pixels. Ignored is `p` is not None.
    palette : list of strings of hex colors, or since hex string
        If a list, color palette to use. If a single string representing
        a hex color, all glyphs are colored with that color. Default is
        the default color cycle employed by HoloViews.
    show_legend : bool, default False
        If True, show legend.
    formal : bool, default False
        If True, make a plot of a formal ECDF (staircase). If False,
        plot the ECDF as dots.
    order : list or None
        If not None, must be a list of unique entries in `df[val]`. The
        order of the list specifies the order of the boxes. If None,
        the boxes appear in the order in which they appeared in the
        inputted DataFrame.
    x_axis_type : 'linear' or 'log'
        Type of x-axis.
    kwargs
        Any kwargs to be passed to p.circle when making the ECDF.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with ECDFs.
    """
    if x_axis_label is None:
        x_axis_label = val
    if y_axis_label is None:
        y_axis_label = 'ECDF'

    return _catplot(df,
                    cats, 
                    val, 
                    'ecdf', 
                    p=p, 
                    x_axis_label=x_axis_label,
                    y_axis_label=y_axis_label,
                    title=title,
                    plot_height=plot_height, 
                    plot_width=plot_width, 
                    palette=palette,
                    show_legend=show_legend,
                    formal=formal,
                    order=order, 
                    x_axis_type=x_axis_type,
                    **kwargs)


def bokeh_colored_ecdf(
        df, cats, val, p=None, x_axis_label=None, y_axis_label=None,
        title=None, plot_height=300, plot_width=400, 
        palette=bokeh.palettes.d3['Category10'][10],
        show_legend=True, order=None, x_axis_type='linear', **kwargs):
    """
    Make an ECDF where points are colored by categorial variables.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame containing tidy data for plotting.
    cats : hashable or list of hastables
        Name of column(s) to use as categorical variable (x-axis). This is
        akin to a kdim in HoloViews.
    val : hashable
        Name of column to use as value variable. This is akin to a vdim
        in HoloViews.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    x_axis_label : str, default None
        Label for the x-axis. Ignored is `p` is not None.
    y_axis_label : str, default 'ECDF'
        Label for the y-axis. Ignored is `p` is not None.
    title : str, default None
        Title of the plot. Ignored is `p` is not None.
    plot_height : int, default 300
        Height of plot, in pixels. Ignored is `p` is not None.
    plot_width : int, default 450
        Width of plot, in pixels. Ignored is `p` is not None.
    palette : list of strings of hex colors, or since hex string
        If a list, color palette to use. If a single string representing
        a hex color, all glyphs are colored with that color. Default is
        the default color cycle employed by HoloViews.
    show_legend : bool, default False
        If True, show legend.
    order : list or None
        If not None, must be a list of unique entries in `df[val]`. The
        order of the list specifies the order of the boxes. If None,
        the boxes appear in the order in which they appeared in the
        inputted DataFrame.
    x_axis_type : 'linear' or 'log'
        Type of x-axis.
    kwargs
        Any kwargs to be passed to p.circle when making the ECDF.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with a colored ECDF.
    """
    if x_axis_label is None:
        x_axis_label = val
    if y_axis_label is None:
        y_axis_label = 'ECDF'
    if 'formal' in kwargs:
        raise RuntimeError('`formal` kwarg not allowed for colored ECDF.')

    return _catplot(df,
                    cats, 
                    val, 
                    'colored_ecdf', 
                    p=p, 
                    x_axis_label=x_axis_label,
                    y_axis_label=y_axis_label,
                    title=title,
                    plot_height=plot_height, 
                    plot_width=plot_width, 
                    palette=palette,
                    show_legend=show_legend,
                    formal=False,
                    order=order, 
                    x_axis_type=x_axis_type,
                    **kwargs)


def bokeh_jitter(df, cats, val, p=None, x_axis_label=None, y_axis_label=None, 
                 title=None, plot_height=300, plot_width=400, 
                 palette=bokeh.palettes.d3['Category10'][10],
                 jitter_width=0.5, order=None, **kwargs):
    """
    Make a jitter plot from a tidy DataFrame.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame containing tidy data for plotting.
    cats : hashable or list of hastables
        Name of column(s) to use as categorical variable (x-axis). This is
        akin to a kdim in HoloViews.
    val : hashable
        Name of column to use as value variable. This is akin to a vdim
        in HoloViews.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    x_axis_label : str, default None
        Label for the x-axis. Ignored is `p` is not None.
    y_axis_label : str, default 'ECDF'
        Label for the y-axis. Ignored is `p` is not None.
    title : str, default None
        Title of the plot. Ignored is `p` is not None.
    plot_height : int, default 300
        Height of plot, in pixels. Ignored is `p` is not None.
    plot_width : int, default 450
        Width of plot, in pixels. Ignored is `p` is not None.
    palette : list of strings of hex colors, or since hex string
        If a list, color palette to use. If a single string representing
        a hex color, all glyphs are colored with that color. Default is
        the default color cycle employed by HoloViews.
    jitter_width : float, default 0.5
        Maximum allowable width of jittered points. A value of 1 means
        that the points take the entire space allotted.
    order : list or None
        If not None, must be a list of unique entries in `df[val]`. The
        order of the list specifies the order of the boxes. If None,
        the boxes appear in the order in which they appeared in the
        inputted DataFrame.
    kwargs
        Any kwargs to be passed to p.circle when making the jitter plot.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with jitter plot.
    """
    return _catplot(df,
                    cats, 
                    val, 
                    'jitter', 
                    p=p, 
                    x_axis_label=x_axis_label,
                    y_axis_label=y_axis_label,
                    title=title,
                    plot_height=plot_height, 
                    plot_width=plot_width, 
                    palette=palette, 
                    width=jitter_width, 
                    show_legend=False,
                    order=order, 
                    **kwargs)


def bokeh_box(df, cats, val, p=None, x_axis_label=None, y_axis_label=None, 
              title=None, plot_height=300, plot_width=400, 
              palette=bokeh.palettes.d3['Category10'][10],
              box_width=0.5, order=None, **kwargs):
    """
    Make a box-and-whisker plot from a tidy DataFrame.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame containing tidy data for plotting.
    cats : hashable or list of hastables
        Name of column(s) to use as categorical variable (x-axis). This is
        akin to a kdim in HoloViews.
    val : hashable
        Name of column to use as value variable. This is akin to a vdim
        in HoloViews.
    p : bokeh.plotting.Figure instance, or None (default)
        If None, create a new figure. Otherwise, populate the existing
        figure `p`.
    x_axis_label : str, default None
        Label for the x-axis. Ignored is `p` is not None.
    y_axis_label : str, default 'ECDF'
        Label for the y-axis. Ignored is `p` is not None.
    title : str, default None
        Title of the plot. Ignored is `p` is not None.
    plot_height : int, default 300
        Height of plot, in pixels. Ignored is `p` is not None.
    plot_width : int, default 450
        Width of plot, in pixels. Ignored is `p` is not None.
    palette : list of strings of hex colors, or since hex string
        If a list, color palette to use. If a single string representing
        a hex color, all boxes are colored with that color. Default is
        the default color cycle employed by HoloViews.
    box_width : float, default 0.5
        Maximum allowable width of the boxes. A value of 1 means that
        the boxes take the entire space allotted.
    order : list or None
        If not None, must be a list of unique entries in `df[val]`. The
        order of the list specifies the order of the boxes. If None,
        the boxes appear in the order in which they appeared in the
        inputted DataFrame.
    kwargs
        Any kwargs to be passed to p.quad when making the plot.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Plot populated with box-and-whisker plot.

    Notes
    -----
    .. Uses the Tukey convention for box plots. The top and bottom of
       the box are respectively the 75th and 25th percentiles of the
       data. The line in the middle of the box is the median. The 
       top whisker extends to the lesser of the largest data point and
       the top of the box plus 1.5 times the interquartile region (the
       height of the box). The bottom whisker extends to the greater of 
       the smallest data point and the bottom of the box minus 1.5 times
       the interquartile region. Data points not between the ends of the
       whiskers are considered outliers and are plotted as individual
       points.
    """
    return _catplot(df,
                    cats, 
                    val, 
                    'box', 
                    p=p, 
                    x_axis_label=x_axis_label,
                    y_axis_label=y_axis_label,
                    title=title,
                    plot_height=plot_height, 
                    plot_width=plot_width, 
                    palette=palette,
                    width=box_width, 
                    show_legend=False,
                    order=order, 
                    **kwargs)


def _display_clicks(div, attributes=[],
                    style='float:left;clear:left;font_size=0.5pt'):
    """Build a suitable CustomJS to display the current event
    in the div model."""
    return bokeh.models.CustomJS(args=dict(div=div), code="""
        var attrs = %s; var args = [];
        for (var i=0; i<attrs.length; i++ ) {
            args.push(Number(cb_obj[attrs[i]]).toFixed(4));
        }
        var line = "<span style=%r>[" + args.join(", ") + "], </span>\\n";
        var text = div.text.concat(line);
        var lines = text.split("\\n")
        if ( lines.length > 35 ) { lines.shift(); }
        div.text = lines.join("\\n");
    """ % (attributes, style))


def bokeh_imshow(im, color_mapper=None, plot_height=400, plot_width=None,
                 length_units='pixels', interpixel_distance=1.0,
                 x_range=None, y_range=None, colorbar=False,
                 no_ticks=False, x_axis_label=None, y_axis_label=None, 
                 title=None, flip=True, return_im=False,
                 saturate_channels=True, min_intensity=None,
                 max_intensity=None, record_clicks=False):
    """
    Display an image in a Bokeh figure.
    
    Parameters
    ----------
    im : Numpy array
        If 2D, intensity image to be displayed. If 3D, first two
        dimensions are pixel values. Last dimension can be of length
        1, 2, or 3, which specify colors.
    color_mapper : str or bokeh.models.LinearColorMapper, default None
        If `im` is an intensity image, `color_mapper` is a mapping of 
        intensity to color. If None, default is 256-level Viridis.
        If `im` is a color image, then `color_mapper` can either be
        'rgb' or 'cmy' (default), for RGB or CMY merge of channels.
    plot_height : int
        Height of the plot in pixels. The width is scaled so that the 
        x and y distance between pixels is the same.
    length_units : str, default 'pixels'
        The units of length in the image.
    interpixel_distance : float, default 1.0
        Interpixel distance in units of `length_units`.
    x_range : bokeh.models.Range1d instance, default None
        Range of x-axis. If None, determined automatically.
    y_range : bokeh.models.Range1d instance, default None
        Range of y-axis. If None, determined automatically.
    colorbar : bool, default False
        If True, include a colorbar.
    no_ticks : bool, default False
        If True, no ticks are displayed. See note below.
    flip : bool, default True
        If True, flip image so it displays right-side up. This is
        necessary because traditionally images have their 0,0 pixel
        index in the top left corner, and not the bottom left corner.
    return_im : bool, default False
        If True, return the GlyphRenderer instance of the image being
        displayed.
    min_intensity : int or float, default None
        Minimum possible intensity of a pixel in the image. If None,
        the image is scaled based on the dynamic range in the image.
    max_intensity : int or float, default None
        Maximum possible intensity of a pixel in the image. If None,
        the image is scaled based on the dynamic range in the image.
    record_clicks : bool, default False
        If True, enables recording of clicks on the image. The clicks are
        displayed in copy-able text next to the displayed figure. 
        
    Returns
    -------
    p : bokeh.plotting.figure instance
        Bokeh plot with image displayed.
    im : bokeh.models.renderers.GlyphRenderer instance (optional)
        The GlyphRenderer instance of the image being displayed. This is
        only returned if `return_im` is True. 

    Notes
    -----
    .. The plot area is set to closely approximate square pixels, but
       this is not always possible since Bokeh sets the plotting area
       based on the entire plot, inclusive of ticks and titles. However,
       if you choose `no_ticks` to be True, no tick or axes labels are
       present, and the pixels are displayed as square.
    """
    # If a single channel in 3D image, flatten and check shape
    if im.ndim == 3:
        if im.shape[2] == 1:
            im = im[:,:,0]
        elif im.shape[2] not in [2, 3]:
            raise RuntimeError('Can only display 1, 2, or 3 channels.')

    # If binary image, make sure it's int
    if im.dtype == bool:
        im = im.astype(np.uint8)

    # Get color mapper
    if im.ndim == 2:
        if color_mapper is None:
            color_mapper = bokeh.models.LinearColorMapper(
                                        bokeh.palettes.viridis(256))
        elif (type(color_mapper) == str 
                and color_mapper.lower() in ['rgb', 'cmy']):
            raise RuntimeError(
                    'Cannot use rgb or cmy colormap for intensity image.')
        if min_intensity is None:
            color_mapper.low = im.min()
        else:
            color_mapper.low = min_intensity
        if max_intensity is None:
            color_mapper.high = im.max()
        else:
            color_mapper.high = max_intensity
    elif im.ndim == 3:
        if color_mapper is None or color_mapper.lower() == 'cmy':
            im = im_merge(*np.rollaxis(im, 2),
                          cmy=True, 
                          im_0_min=min_intensity,
                          im_1_min=min_intensity,
                          im_2_min=min_intensity,
                          im_0_max=max_intensity,
                          im_1_max=max_intensity,
                          im_2_max=max_intensity)
        elif color_mapper.lower() == 'rgb':
            im = im_merge(*np.rollaxis(im, 2),
                          cmy=False, 
                          im_0_min=min_intensity,
                          im_1_min=min_intensity,
                          im_2_min=min_intensity,
                          im_0_max=max_intensity,
                          im_1_max=max_intensity,
                          im_2_max=max_intensity)
        else:
            raise RuntimeError('Invalid color mapper for color image.')
    else:
        raise RuntimeError(
                    'Input image array must have either 2 or 3 dimensions.')

    # Get shape, dimensions
    n, m = im.shape[:2]
    if x_range is not None and y_range is not None:
        dw = x_range[1] - x_range[0]
        dh = y_range[1] - y_range[0]
    else:
        dw = m * interpixel_distance
        dh = n * interpixel_distance
        x_range = [0, dw]
        y_range = [0, dh]
    
    # Set up figure with appropriate dimensions
    if plot_width is None:
        plot_width = int(m/n * plot_height)
    if colorbar:
        plot_width += 40
        toolbar_location = 'above'
    else:
        toolbar_location = 'right'
    p = bokeh.plotting.figure(plot_height=plot_height,
                              plot_width=plot_width,
                              x_range=x_range,
                              y_range=y_range,
                              title=title,
                              toolbar_location=toolbar_location,
                              tools='pan,box_zoom,wheel_zoom,reset')
    if no_ticks:
        p.xaxis.major_label_text_font_size = '0pt'
        p.yaxis.major_label_text_font_size = '0pt'
        p.xaxis.major_tick_line_color = None 
        p.xaxis.minor_tick_line_color = None
        p.yaxis.major_tick_line_color = None 
        p.yaxis.minor_tick_line_color = None
    else:
        if x_axis_label is None:
            p.xaxis.axis_label = length_units
        else:
            p.xaxis.axis_label = x_axis_label
        if y_axis_label is None:
            p.yaxis.axis_label = length_units
        else:
            p.yaxis.axis_label = y_axis_label

    # Display the image
    if im.ndim == 2:
        if flip:
            im = im[::-1,:]
        im_bokeh = p.image(image=[im],
                           x=x_range[0], 
                           y=y_range[0], 
                           dw=dw, 
                           dh=dh, 
                           color_mapper=color_mapper)
    else:
        im_bokeh = p.image_rgba(image=[rgb_to_rgba32(im, flip=flip)], 
                                x=x_range[0],
                                y=y_range[0],
                                dw=dw, 
                                dh=dh)

    # Make a colorbar
    if colorbar:
        if im.ndim == 3:
            warnings.warn('No colorbar display for RGB images.')
        else:
            color_bar = bokeh.models.ColorBar(color_mapper=color_mapper,
                                              label_standoff=12,
                                              border_line_color=None,
                                              location=(0,0))
            p.add_layout(color_bar, 'right')

    if record_clicks:
        div = bokeh.models.Div(width=200)
        layout = bokeh.layouts.row(p, div)
        p.js_on_event(bokeh.events.Tap,
                      _display_clicks(div, attributes=['x', 'y']))
        if return_im:
            return layout, im_bokeh
        else:
            return layout

    if return_im:
        return p, im_bokeh
    return p


def im_merge(im_0, im_1, im_2=None, im_0_max=None,
             im_1_max=None, im_2_max=None, im_0_min=None,
             im_1_min=None, im_2_min=None, cmy=True):
    """
    Merge channels to make RGB image.

    Parameters
    ----------
    im_0: array_like
        Image represented in first channel.  Must be same shape
        as `im_1` and `im_2` (if not None).
    im_1: array_like
        Image represented in second channel.  Must be same shape
        as `im_1` and `im_2` (if not None).
    im_2: array_like, default None
        Image represented in third channel.  If not None, must be same
        shape as `im_0` and `im_1`.
    im_0_max : float, default max of inputed first channel
        Maximum value to use when scaling the first channel. If None,
        scaled to span entire range.
    im_1_max : float, default max of inputed second channel
        Maximum value to use when scaling the second channel
    im_2_max : float, default max of inputed third channel
        Maximum value to use when scaling the third channel
    im_0_min : float, default min of inputed first channel
        Maximum value to use when scaling the first channel
    im_1_min : float, default min of inputed second channel
        Minimum value to use when scaling the second channel
    im_2_min : float, default min of inputed third channel
        Minimum value to use when scaling the third channel
    cmy : bool, default True
        If True, first channel is cyan, second is magenta, and third is
        yellow. Otherwise, first channel is red, second is green, and 
        third is blue.

    Returns
    -------
    output : array_like, dtype float, shape (*im_0.shape, 3)
        RGB image.
    """

    # Compute max intensities if needed
    if im_0_max is None:
        im_0_max = im_0.max()
    if im_1_max is None:
        im_1_max = im_1.max()
    if im_2 is not None and im_2_max is None:
        im_2_max = im_2.max()

    # Compute min intensities if needed
    if im_0_min is None:
        im_0_min = im_0.min()
    if im_1_min is None:
        im_1_min = im_1.min()
    if im_2 is not None and im_2_min is None:
        im_2_min = im_2.min()

    # Make sure maxes are ok
    if im_0_max < im_0.max() or im_1_max < im_1.max() \
            or (im_2 is not None and im_2_max < im_2.max()):
        raise RuntimeError(
                'Inputted max of channel < max of inputted channel.')

    # Make sure mins are ok
    if im_0_min > im_0.min() or im_1_min > im_1.min() \
            or (im_2 is not None and im_2_min > im_2.min()):
        raise RuntimeError(
                'Inputted min of channel > min of inputted channel.')

    # Scale the images
    if im_0_max > im_0_min:
        im_0 = (im_0 - im_0_min) / (im_0_max - im_0_min)
    else:
        im_0 = (im_0 > 0).astype(float)

    if im_1_max > im_1_min:
        im_1 = (im_1 - im_1_min) / (im_1_max - im_1_min)
    else:
        im_0 = (im_0 > 0).astype(float)

    if im_2 is None:
        im_2 = np.zeros_like(im_0)
    elif im_2_max > im_2_min:
        im_2 = (im_2 - im_2_min) / (im_2_max - im_2_min)
    else:
        im_0 = (im_0 > 0).astype(float)

    # Convert images to RGB
    if cmy:
        im_c = np.stack((np.zeros_like(im_0), im_0, im_0), axis=2)
        im_m = np.stack((im_1, np.zeros_like(im_1), im_1), axis=2)
        im_y = np.stack((im_2, im_2, np.zeros_like(im_2)), axis=2)
        im_rgb = im_c + im_m + im_y
        for i in [0, 1, 2]:
            im_rgb[:,:,i] /= im_rgb[:,:,i].max()
    else:
        im_rgb = np.empty((*im_0.shape, 3))
        im_rgb[:,:,0] = im_0
        im_rgb[:,:,1] = im_1
        im_rgb[:,:,2] = im_2

    return im_rgb


def rgb_to_rgba32(im, flip=True):
    """
    Convert an RGB image to a 32 bit-encoded RGBA image.

    Parameters
    ----------
    im : ndarray, shape (nrows, ncolums, 3)
        Input image. All pixel values must be between 0 and 1.
    flip : bool, default True
        If True, flip image so it displays right-side up. This is
        necessary because traditionally images have their 0,0 pixel
        index in the top left corner, and not the bottom left corner.

    Returns
    -------
    output : ndarray, shape (nros, ncolumns), dtype np.uint32
        Image decoded as a 32 bit RBGA image.
    """
    # Ensure it has three channels
    if im.ndim != 3 or im.shape[2] !=3:
        raise RuntimeError('Input image is not RGB.')

    # Make sure all entries between zero and one
    if (im < 0).any() or (im > 1).any():
        raise RuntimeError('All pixel values must be between 0 and 1.')

    # Get image shape
    n, m, _ = im.shape

    # Convert to 8-bit, which is expected for viewing
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        im_8 = skimage.img_as_ubyte(im)

    # Add the alpha channel, which is expected by Bokeh
    im_rgba = np.stack((*np.rollaxis(im_8, 2),
                        255*np.ones((n, m), dtype=np.uint8)), axis=2)

    # Reshape into 32 bit. Must flip up/down for proper orientation
    if flip:
        return np.flipud(im_rgba.view(dtype=np.int32).reshape((n, m)))
    else:
        return im_rgba.view(dtype=np.int32).reshape((n, m))


def rgb_frac_to_hex(rgb_frac):
    """
    Convert fractional RGB values to hexadecimal color string.

    Parameters
    ----------
    rgb_frac : array_like, shape (3,)
        Fractional RGB values; each entry is between 0 and 1.

    Returns
    -------
    str
        Hexidecimal string for the given RGB color.

    Examples
    --------
    >>> rgb_frac_to_hex((0.65, 0.23, 1.0))
    '#a53aff'

    >>> rgb_frac_to_hex((1.0, 1.0, 1.0))
    '#ffffff'
    """

    if len(rgb_frac) != 3:
        raise RuntimeError('`rgb_frac` must have exactly three entries.')

    if (np.array(rgb_frac) < 0).any() or (np.array(rgb_frac) > 1).any():
        raise RuntimeError('RGB values must be between 0 and 1.')

    return '#{0:02x}{1:02x}{2:02x}'.format(int(rgb_frac[0] * 255),
                                           int(rgb_frac[1] * 255),
                                           int(rgb_frac[2] * 255))


def bokeh_contour(X, Y, Z, levels=None, p=None, overlaid=False,
                  plot_width=350, plot_height=300, x_axis_label='x',
                  y_axis_label='y', title=None, line_color=None, line_width=2,
                  color_mapper=None, overlay_grid=False, fill=False, 
                  fill_palette=None, fill_alpha=0.75, **kwargs):
    """
    Make a contour plot, possibly overlaid on an image.

    Parameters
    ----------
    X : 2D Numpy array
        Array of x-values, as would be produced using np.meshgrid()
    Y : 2D Numpy array
        Array of y-values, as would be produced using np.meshgrid()
    Z : 2D Numpy array
        Array of z-values.
    levels : array_like
        Levels to plot, ranging from 0 to 1. The contour around a given
        level contains that fraction of the total probability if the
        contour plot is for a 2D probability density function. By 
        default, the levels are given by the one, two, three, and four
        sigma levels corresponding to a marginalized distribution from
        a 2D Gaussian distribution.
    p : bokeh plotting object, default None
        If not None, the contour are added to `p`. This option is not
        allowed if `overlaid` is True.
    overlaid : bool, default False
        If True, `Z` is displayed as an image and the contours are
        overlaid.
    plot_width : int, default 350
        Width of the plot in pixels. Ignored if `p` is not None.
    plot_height : int, default 300
        Height of the plot in pixels. Ignored if `p` is not None.
    x_axis_label : str, default 'x'
        Label for the x-axis. Ignored if `p` is not None.
    y_axis_label : str, default 'y'
        Label for the y-axis. Ignored if `p` is not None.
    title : str, default None
        Title of the plot. Ignored if `p` is not None.
    line_color : str, defaults to Bokeh default
        Color, either named CSS color or hex, of contour lines.
    line_width : int, default 2
        Width of contour lines.
    color_mapper : bokeh.models.LinearColorMapper, default Viridis
        Mapping of `Z` level to color. Ignored if `overlaid` is False.
    overlay_grid : bool, default False
        If True, faintly overlay the grid on top of image. Ignored if
        overlaid is False.

    Returns
    -------
    output : Bokeh plotting object
        Plot populated with contours, possible with an image.
    """
    if len(X.shape) != 2 or Y.shape != X.shape or Z.shape != X.shape:
        raise RuntimeError('All arrays must be 2D and of same shape.')

    if overlaid and p is not None:
        raise RuntimeError('Cannot specify `p` if showing image.')

    if line_color is None:
        if overlaid:
            line_color = 'white'
        else:
            line_color = 'black'

    if p is None:
        if overlaid:
            p = bokeh_imshow(Z,
                       color_mapper=color_mapper,
                       plot_height=plot_height,
                       plot_width=plot_width,
                       x_axis_label=x_axis_label,
                       y_axis_label=y_axis_label,
                       title=title,
                       x_range = [X.min(), X.max()],
                       y_range = [Y.min(), Y.max()],
                       no_ticks=False, 
                       flip=False, 
                       return_im=False)
        else:
            p = bokeh.plotting.figure(plot_width=plot_width,
                                      plot_height=plot_height,
                                      x_axis_label=x_axis_label,
                                      y_axis_label=y_axis_label,
                                      title=title)

    # Set default levels
    if levels is None:
        levels = 1.0 - np.exp(-np.arange(0.5, 2.1, 0.5)**2 / 2)

    # Compute contour lines
    if fill or line_width:
        xs, ys = _contour_lines(X, Y, Z, levels)

    # Make fills. This is currently not supported
    if fill:
        raise NotImplementedError('Filled contours are not yet implemented.')
        if fill_palette is None:
            if len(levels) <= 6:
                fill_palette = bokeh.palettes.Greys[len(levels)+3][1:-1]
            elif len(levels) <= 10:
                fill_palette = bokeh.palettes.Viridis[len(levels)+1]
            else:
                raise RuntimeError(
                    'Can only have maximally 10 levels with filled contours' +
                    ' unless user specifies `fill_palette`.')
        elif len(fill_palette) != len(levels) + 1:
            raise RuntimeError('`fill_palette` must have 1 more entry' +
                               ' than `levels`')

        p.patch(xs[-1], ys[-1],
                color=fill_palette[0],
                alpha=fill_alpha,
                line_color=None)
        for i in range(1, len(levels)):
            x_p = np.concatenate((xs[-1-i], xs[-i][::-1]))
            y_p = np.concatenate((ys[-1-i], ys[-i][::-1]))
            print(len(x_p), len(y_p))
            p.patch(x_p, 
                    y_p, 
                    color=fill_palette[i],
                    alpha=fill_alpha,
                    line_color=None)

        p.background_fill_color=fill_palette[-1]

    # Populate the plot with contour lines
    if line_width:
        p.multi_line(xs, ys, line_color=line_color, line_width=line_width,
                     **kwargs)

    if overlay_grid and overlaid:
        p.grid.level = 'overlay'
        p.grid.grid_line_alpha = 0.2

    return p


def _data_range(df, x, y, margin=0.02):
    x_range = df[x].max() - df[x].min()
    y_range = df[y].max() - df[y].min()
    return ([df[x].min() - x_range*margin, df[x].max() + x_range*margin],
            [df[y].min() - y_range*margin, df[y].max() + y_range*margin])


def _create_points_image(x_range, y_range, w, h, df, x, y, cmap):
    cvs = ds.Canvas(x_range=x_range, y_range=y_range, plot_height=int(h), 
                    plot_width=int(w))
    agg = cvs.points(df, x, y, agg=ds.reductions.count())
    return ds.transfer_functions.dynspread(ds.transfer_functions.shade(
                                        agg, cmap=cmap, how='linear'))


def _create_line_image(x_range, y_range, w, h, df, x, y, cmap=None):
    cvs = ds.Canvas(x_range=x_range, y_range=y_range, plot_height=int(h), 
                    plot_width=int(w))
    agg = cvs.line(df, x, y)
    return ds.transfer_functions.dynspread(ds.transfer_functions.shade(
                                               agg, cmap=cmap))


def _contour_lines(X, Y, Z, levels):
    """
    Generate lines for contour plot.
    """
    # Compute the density levels.
    Zflat = Z.flatten()
    inds = np.argsort(Zflat)[::-1]
    Zflat = Zflat[inds]
    sm = np.cumsum(Zflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Zflat[sm <= v0][-1]
        except:
            V[i] = Zflat[0]
    V.sort()
    m = np.diff(V) == 0
    
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(V) == 0
    V.sort()

    # Make contours
    c = matplotlib._contour.QuadContourGenerator(X, Y, Z, None, True, 0)
    xs = []
    ys = []
    for level in V:
        paths = c.create_contour(level)
        for line in paths:
            xs.append(line[:,0])
            ys.append(line[:,1])
            
    return xs, ys


def _get_contour_lines_from_samples(x, y, smooth=1, levels=None, bins=50, 
                                    weights=None, extend_domain=False):
    """
    Get lines for contour overlay.

    Based on code from emcee by Dan Forman-Mackey.
    """
    data_range = [[x.min(), x.max()], [y.min(), y.max()]]

    # Choose the default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    # We'll make the 2D histogram to directly estimate the density.
    try:
        H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=bins,
                                 range=list(map(np.sort, data_range)),
                                 weights=weights)
    except ValueError:
        raise ValueError("It looks like at least one of your sample columns "
                         "have no dynamic data_range. You could try using the "
                         "'data_range' argument.")

    if smooth is not None:
        H = scipy.ndimage.gaussian_filter(H, smooth)

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    if extend_domain:
        H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
        H2[2:-2, 2:-2] = H
        H2[2:-2, 1] = H[:, 0]
        H2[2:-2, -2] = H[:, -1]
        H2[1, 2:-2] = H[0]
        H2[-2, 2:-2] = H[-1]
        H2[1, 1] = H[0, 0]
        H2[1, -2] = H[0, -1]
        H2[-2, 1] = H[-1, 0]
        H2[-2, -2] = H[-1, -1]
        X2 = np.concatenate([
            X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
            X1,
            X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
        ])
        Y2 = np.concatenate([
            Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
            Y1,
            Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
        ])
        X2, Y2 = np.meshgrid(X2, Y2)
    else:
        X2, Y2 = np.meshgrid(X1, Y1)
        H2 = H

    return _contour_lines(X2, Y2, H2.transpose(), levels)


def bokeh_im_click(im, color_mapper=None, plot_height=400, plot_width=None,
                   length_units='pixels', interpixel_distance=1.0,
                   x_range=None, y_range=None, no_ticks=False,
                   x_axis_label=None, y_axis_label=None, title=None, 
                   flip=True):
    """
    """

    def display_event(div, attributes=[],
                      style='float:left;clear:left;font_size=0.5pt'):
        """Build a suitable CustomJS to display the current event
        in the div model."""
        return bokeh.models.CustomJS(args=dict(div=div), code="""
            var attrs = %s; var args = [];
            for (var i=0; i<attrs.length; i++ ) {
                args.push(Number(cb_obj[attrs[i]]).toFixed(4));
            }
            var line = "<span style=%r>[" + args.join(", ") + "],</span>\\n";
            var text = div.text.concat(line);
            var lines = text.split("\\n")
            if ( lines.length > 35 ) { lines.shift(); }
            div.text = lines.join("\\n");
        """ % (attributes, style))

    p = bokeh_imshow(im,
               color_mapper=color_mapper,
               plot_height=plot_height, 
               plot_width=plot_width,
               length_units=length_units, 
               interpixel_distance=interpixel_distance,
               x_range=x_range, 
               y_range=y_range,
               no_ticks=no_ticks, 
               x_axis_label=x_axis_label, 
               y_axis_label=y_axis_label, 
               title=title, 
               flip=flip)

    div = bokeh.models.Div(width=200)
    layout = bokeh.layouts.row(p, div)

    p.js_on_event(bokeh.events.Tap, display_event(div, attributes=['x', 'y']))

    return layout


def mpl_cmap_to_bokeh_color_mapper(cmap):
    """
    Convert a Matplotlib colormap to a bokeh.models.LinearColorMapper
    instance.

    Parameters
    ----------
    cmap : str
        A string giving the name of the color map.

    Returns
    -------
    output : bokeh.models.LinearColorMapper instance
        A linear color_mapper with 256 gradations.

    Notes
    -----
    .. See https://matplotlib.org/examples/color/colormaps_reference.html
       for available Matplotlib colormaps.
    """
    cm = mpl_get_cmap(cmap)
    palette = [rgb_frac_to_hex(cm(i)[:3]) for i in range(256)]
    return bokeh.models.LinearColorMapper(palette=palette)


