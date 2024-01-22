/*
*                      *** Used by CI ***
* This file is used by deploy-pr.yml when building the PR Previews for QML.
* This is a modified gatsby-node.ts file that replaces the gatsby-node in pennylane-website build
* and remove the dependency on other SWC backend service that the QML PR Previews does not need.
*/


/* eslint-disable @typescript-eslint/no-var-requires */
const { createFilePath } = require(`gatsby-source-filesystem`)
// Define the template for blog post
import {
  CreateBabelConfigArgs,
  GatsbyNode,
} from 'gatsby'
import { demosCategories } from './content/demos/demonstrations_categories'
import path from 'path'
import { createDemoCategorySearchRoute } from './src/utils/url_helpers'

interface IOnCreateNodeProps {
  node: { internal: { type: string } }
  actions: {
    createNodeField: (field: {
      name: string
      node: { internal: { type: string } }
      value: string
    }) => void
  }
  getNode: () => void
}

interface IOnCreateNodeProps {
  node: { internal: { type: string } }
  actions: {
    createNodeField: (field: {
      name: string
      node: { internal: { type: string } }
      value: string
    }) => void
  }
  getNode: () => void
}

/**
 * @type {import('gatsby').GatsbyNode['onCreateNode']}
 */
exports.onCreateNode = ({ node, actions, getNode }: IOnCreateNodeProps) => {
  const { createNodeField } = actions

  if (node.internal.type === `MarkdownRemark`) {
    const value = createFilePath({ node, getNode })

    createNodeField({
      name: `slug`,
      node,
      value,
    })
  }
}

exports.onCreateBabelConfig = ({ actions }: CreateBabelConfigArgs) => {
  actions.setBabelPlugin({
    name: '@babel/plugin-transform-react-jsx',
    options: {
      runtime: 'automatic',
    },
  })
}

/**
 * @type {import('gatsby').GatsbyNode['createSchemaCustomization']}
 */
exports.createSchemaCustomization = ({
  actions,
}: {
  actions: {
    createTypes: (types: string) => void
  }
}) => {
  const { createTypes } = actions

  // Explicitly define the siteMetadata {} object
  // This way those will always be defined even if removed from gatsby-config.js

  // Also explicitly define the Markdown frontmatter
  // This way the "MarkdownRemark" queries will return `null` even when no
  // blog posts are stored inside "content/blog" instead of returning an error
  createTypes(`
    type SiteSiteMetadata {
      author: Author
      siteUrl: String
      social: Social
    }

    type Author {
      name: String
      summary: String
    }

    type Social {
      twitter: String
    }

    type MarkdownRemark implements Node {
      frontmatter: Frontmatter
      fields: Fields
    }

    type Frontmatter {
      title: String
      description: String
      date: Date @dateformat
      categories: [String!]
      relatedContent: [RelatedContent]
      hardware: [Hardware]
    }

    type Fields {
      slug: String
    }

    type RelatedContent {
      id: String!
      title: String!
      previewImages: [PreviewImages]
    }

    type PreviewImages {
      type: String!
      uri: String!
    }

    type Hardware {
      id: String!
      link: String!
      logo: String!
    }
  `)
}

const demoPage = path.resolve(
  `${__dirname}/src/templates/demos/individualDemo/demo.tsx`
)

export const createPages: GatsbyNode['createPages'] = async ({
  graphql,
  actions,
  reporter,
}) => {
  const { createPage, createRedirect } = actions

  type allMarkdownRemarkTypeData = {
    allMarkdownRemark: {
      nodes: {
        frontmatter: {
          slug: string
          title: string
          meta_description: string
        }
        id: string
      }[]
    }
  }

  const DemosResults = await graphql<allMarkdownRemarkTypeData>(`
    query GetDemoData {
      allMarkdownRemark(filter: { fileAbsolutePath: { regex: "/demos/" } }) {
        nodes {
          frontmatter {
            slug
            title
            meta_description
          }
          id
        }
      }
    }
  `)

  const demos = DemosResults.data
    ? DemosResults.data.allMarkdownRemark.nodes
    : []

  if (demos && demos.length) {
    demos.forEach((demo) => {
      createPage({
        path: `/qml/demos/${demo.frontmatter.slug}`,
        component: demoPage,
        context: {
          id: demo.id,
        },
      })
    })
  }

  /* Redirect from category pages to search page */
  if (demosCategories) {
    demosCategories.forEach((category) => {
      if (category.urlFragment)
        createRedirect({
          fromPath: `/qml/demonstrations/${category.urlFragment}/`,
          toPath: createDemoCategorySearchRoute(category.title),
          isPermanent: true,
          redirectInBrowser: true,
        })
    })
  }
}