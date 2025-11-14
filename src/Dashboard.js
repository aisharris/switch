const Dashboard = (video_bool) => {
  // const [dashboardId, setDashboardId] = useState('');

  const kibanaUrlBase = "http://localhost:5601/app/dashboards#/view/";
  const kibanaUrlQuery = "?embed=true&_g=(filters:!(),refreshInterval:(pause:!t,value:0),time:(from:now%2Fw,to:now%2Fw))";
  
  if (video_bool.video_bool === false) {
    console.log("VIDEO FALSE")
    // Original dashboard for image-based metrics
    const imageDashboardId = "1c4bcb30-01e2-11ee-aba5-bbfb87cb88fa";
    // const permalink = "http://localhost:5601/app/discover#/?_g=(time:(from:now%2Fw,to:now%2Fw))&_a=(columns:!(_source),filters:!(),index:'187488b0-1710-11ee-9137-43d9d1b8ee4d',interval:auto,query:(language:kuery,query:''),sort:!())"
    const imageDashboardUrl = `${kibanaUrlBase}${imageDashboardId}${kibanaUrlQuery}`;
    return (
      <div>
        <h1>Image Processing Dashboard</h1>
        <iframe
          src={imageDashboardUrl}
          title="Kibana Dashboard"
          height="800"
          width="100%"
        ></iframe>
      </div>
    );
  } else {
      console.log("VIDEO TRU")

      // Original dashboard for image-based metrics
      const vidDashboardId = "822fec10-99e3-11f0-89f4-8587cbe3076b";
      const vidDashboardUrl = `${kibanaUrlBase}${vidDashboardId}${kibanaUrlQuery}`;
      return (
        <div>
          <h1>Video Processing Dashboard</h1>
          <iframe
            src={vidDashboardUrl}
            title="Kibana Dashboard"
            height="800"
            width="100%"
          ></iframe>
        </div>
      );
    // // Dashboard for video-based metrics with user input
    // const videoDashboardUrl = dashboardId ? `${kibanaUrlBase}${dashboardId}${kibanaUrlQuery}` : null;
    // return (
    //   <div className="flex flex-col items-center p-4 space-y-4">
    //     <h1 className="text-2xl font-bold text-gray-800">Video Processing Dashboard</h1>
    //     <div className="w-full max-w-2xl">
    //       <p className="text-gray-600 mb-2">
    //         Please enter the ID of your Kibana dashboard for video metrics.
    //         This dashboard should be configured to visualize data from the
    //         <code className="bg-gray-200 rounded p-1 text-sm font-mono mx-1">video_processing_metrics</code> and <code className="bg-gray-200 rounded p-1 text-sm font-mono mx-1">video_adaptation_logs</code> indices.
    //       </p>
    //       <div className="flex w-full space-x-2">
    //         <input
    //           type="text"
    //           className="flex-grow p-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
    //           placeholder="Paste Kibana Dashboard ID here..."
    //           value={dashboardId}
    //           onChange={(e) => setDashboardId(e.target.value)}
    //         />
    //       </div>
    //     </div>
    //     {videoDashboardUrl ? (
    //       <iframe
    //         src={videoDashboardUrl}
    //         title="Kibana Video Dashboard"
    //         height="800"
    //         width="100%"
    //         className="border-none shadow-lg rounded-md"
    //       ></iframe>
    //     ) : (
    //       <div className="w-full max-w-2xl p-6 bg-white rounded-lg shadow-md text-center text-gray-500">
    //         Enter a Kibana Dashboard ID to see the visualization.
    //       </div>
    //     )}
    //   </div>
    // );
  }
};

export default Dashboard;
